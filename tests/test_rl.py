"""
Tests for the RL module (Phase 3).
"""
import pytest
import numpy as np
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rl.environment import TradingEnv, EnvConfig
from src.rl.rewards import DifferentialSharpeRatio, RewardConfig, AsymmetricDSR
from src.rl.popart import PopArtNormalizer, PopArtLayer
from src.rl.agent import QRDQNAgent, QRDQNConfig, QRDQNNetwork, ReplayBuffer


class TestTradingEnv:
    """Test Trading Environment."""
    
    @pytest.fixture
    def env(self):
        config = EnvConfig(
            context_dim=64,
            initial_balance=10000,
            max_steps=100,
        )
        # Synthetic price data
        price_data = 50000 + np.cumsum(np.random.randn(1000) * 100)
        return TradingEnv(config, price_data=price_data)
    
    def test_reset(self, env):
        state, info = env.reset()
        
        assert state.shape == (env.state_dim,)
        assert info["balance"] == 10000
        assert info["position"] == 0
    
    def test_step(self, env):
        env.reset()
        
        # Take a buy action
        next_state, reward, terminated, truncated, info = env.step(1)  # Buy
        
        assert next_state.shape == (env.state_dim,)
        assert isinstance(reward, float)
        assert info["position"] > 0  # Should have long position
    
    def test_episode(self, env):
        state, _ = env.reset()
        total_reward = 0
        
        for _ in range(50):
            action = np.random.randint(3)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        assert isinstance(total_reward, float)
    
    def test_action_space(self, env):
        assert env.action_space.n == 3  # Hold, Buy, Sell
    
    def test_position_limits(self, env):
        env.reset()
        
        # Max out position
        for _ in range(10):
            env.step(1)  # Buy
        
        assert env.position <= env.config.max_position


class TestDSR:
    """Test Differential Sharpe Ratio."""
    
    def test_initialization(self):
        dsr = DifferentialSharpeRatio()
        
        assert dsr.A == 0
        assert dsr.B > 0  # Small positive to avoid div zero
        assert dsr.step_count == 0
    
    def test_compute(self):
        dsr = DifferentialSharpeRatio()
        
        # Positive return
        reward1 = dsr.compute(step_return=0.01)
        
        # Another positive return
        reward2 = dsr.compute(step_return=0.02)
        
        assert isinstance(reward1, float)
        assert isinstance(reward2, float)
    
    def test_reset(self):
        dsr = DifferentialSharpeRatio()
        
        dsr.compute(step_return=0.01)
        dsr.compute(step_return=0.02)
        
        dsr.reset()
        
        assert dsr.A == 0
        assert dsr.step_count == 0
    
    def test_risk_penalty(self):
        config = RewardConfig(max_drawdown_penalty=1.0)
        dsr = DifferentialSharpeRatio(config)
        
        # No drawdown
        reward1 = dsr.compute(step_return=0.01, drawdown=0)
        
        dsr.reset()
        
        # With drawdown
        reward2 = dsr.compute(step_return=0.01, drawdown=0.1)
        
        # Reward should be lower with drawdown
        assert reward1 > reward2
    
    def test_asymmetric_dsr(self):
        dsr = AsymmetricDSR(loss_multiplier=2.0)
        
        # Positive return
        reward1 = dsr.compute(step_return=0.01)
        
        dsr.reset()
        
        # Negative return (should be penalized more)
        reward2 = dsr.compute(step_return=-0.01)
        
        # Loss should hurt more than gain helps
        assert abs(reward2) > abs(reward1)


class TestPopArt:
    """Test PopArt normalization."""
    
    def test_normalizer(self):
        normalizer = PopArtNormalizer(input_dim=64, output_dim=3)
        
        x = torch.randn(4, 64)
        output = normalizer(x)
        
        assert output.shape == (4, 3)
    
    def test_denormalize(self):
        normalizer = PopArtNormalizer(input_dim=64, output_dim=1)
        
        # Update stats
        targets = torch.randn(100) * 10 + 5
        normalizer.update_stats(targets.unsqueeze(-1))
        
        x = torch.randn(4, 64)
        normalized = normalizer(x)
        denormalized = normalizer.denormalize(normalized)
        
        # Should be in different ranges
        assert not torch.allclose(normalized, denormalized)
    
    def test_update_preserves_output(self):
        normalizer = PopArtNormalizer(input_dim=32, output_dim=1)
        
        x = torch.randn(4, 32)
        
        # Get output before update
        with torch.no_grad():
            output_before = normalizer.denormalize(normalizer(x))
        
        # Update stats
        targets = torch.randn(100) * 10
        normalizer.update_stats(targets.unsqueeze(-1))
        
        # Get output after update
        with torch.no_grad():
            output_after = normalizer.denormalize(normalizer(x))
        
        # Outputs should be close (PopArt preserves)
        assert torch.allclose(output_before, output_after, rtol=0.1, atol=0.5)
    
    def test_popart_layer(self):
        layer = PopArtLayer(
            input_dim=64,
            num_actions=3,
            num_quantiles=32,
        )
        
        x = torch.randn(4, 64)
        quantiles = layer(x)
        
        assert quantiles.shape == (4, 3, 32)
        
        q_values = layer.get_q_values(x)
        assert q_values.shape == (4, 3)


class TestQRDQN:
    """Test QR-DQN Agent."""
    
    @pytest.fixture
    def config(self):
        return QRDQNConfig(
            state_dim=80,  # 64 context + 16 account
            hidden_dims=[64, 64],
            num_actions=3,
            num_quantiles=16,
            buffer_size=1000,
            min_buffer_size=100,
            batch_size=32,
            device="cpu",
        )
    
    @pytest.fixture
    def network(self, config):
        return QRDQNNetwork(config)
    
    @pytest.fixture
    def agent(self, config):
        return QRDQNAgent(config)
    
    def test_network_forward(self, network):
        x = torch.randn(4, 80)
        quantiles = network(x)
        
        assert quantiles.shape == (4, 3, 16)  # batch, actions, quantiles
    
    def test_network_q_values(self, network):
        x = torch.randn(4, 80)
        q_values = network.get_q_values(x)
        
        assert q_values.shape == (4, 3)
    
    def test_network_action_selection(self, network):
        x = torch.randn(1, 80)
        
        # Greedy
        action = network.get_action(x, epsilon=0.0)
        assert 0 <= action < 3
        
        # With exploration
        action = network.get_action(x, epsilon=1.0)
        assert 0 <= action < 3
    
    def test_agent_select_action(self, agent):
        state = np.random.randn(80).astype(np.float32)
        action = agent.select_action(state)
        
        assert 0 <= action < 3
    
    def test_replay_buffer(self, config):
        buffer = ReplayBuffer(capacity=100, state_dim=config.state_dim)
        
        # Add some experiences
        for _ in range(50):
            state = np.random.randn(config.state_dim).astype(np.float32)
            next_state = np.random.randn(config.state_dim).astype(np.float32)
            buffer.push(state, np.random.randint(3), 0.1, next_state, False)
        
        assert len(buffer) == 50
        
        # Sample
        states, actions, rewards, next_states, dones = buffer.sample(16)
        
        assert states.shape == (16, config.state_dim)
        assert actions.shape == (16, 1)
    
    def test_agent_train_step(self, agent, config):
        # Fill buffer
        for _ in range(config.min_buffer_size + 10):
            state = np.random.randn(config.state_dim).astype(np.float32)
            next_state = np.random.randn(config.state_dim).astype(np.float32)
            agent.store_transition(state, np.random.randint(3), 0.1, next_state, False)
        
        # Train
        metrics = agent.train_step()
        
        assert "loss" in metrics
        assert metrics["loss"] > 0
    
    def test_agent_save_load(self, agent, config, tmp_path):
        # Fill buffer and train
        for _ in range(config.min_buffer_size + 10):
            state = np.random.randn(config.state_dim).astype(np.float32)
            next_state = np.random.randn(config.state_dim).astype(np.float32)
            agent.store_transition(state, np.random.randint(3), 0.1, next_state, False)
        
        agent.train_step()
        
        # Save
        path = tmp_path / "agent.pt"
        agent.save(str(path))
        
        # Load into new agent
        new_agent = QRDQNAgent(config)
        new_agent.load(str(path))
        
        # Check model weights are the same
        for (name1, p1), (name2, p2) in zip(
            agent.online_net.named_parameters(),
            new_agent.online_net.named_parameters()
        ):
            assert torch.allclose(p1, p2), f"Mismatch in {name1}"
        
        # Check train steps match
        assert agent.train_steps == new_agent.train_steps


class TestIntegration:
    """Integration tests for full RL pipeline."""
    
    def test_env_agent_loop(self):
        # Create env
        env_config = EnvConfig(
            context_dim=64,
            initial_balance=10000,
            max_steps=100,
        )
        price_data = 50000 + np.cumsum(np.random.randn(500) * 100)
        env = TradingEnv(env_config, price_data=price_data)
        
        # Create agent
        agent_config = QRDQNConfig(
            state_dim=env.state_dim,
            hidden_dims=[64, 64],
            buffer_size=1000,
            min_buffer_size=100,
            batch_size=32,
            device="cpu",
        )
        agent = QRDQNAgent(agent_config)
        
        # Create reward function
        dsr = DifferentialSharpeRatio()
        
        # Run episode
        state, _ = env.reset()
        dsr.reset()
        
        total_reward = 0
        for step in range(100):
            action = agent.select_action(state)
            next_state, raw_reward, terminated, truncated, info = env.step(action)
            
            # Compute DSR reward
            reward = dsr.compute(
                step_return=info.get("step_return", raw_reward),
                position=env.position,
                drawdown=info.get("drawdown", 0),
            )
            
            agent.store_transition(state, action, reward, next_state, terminated or truncated)
            
            # Train if buffer is ready
            if len(agent.buffer) >= agent_config.min_buffer_size:
                agent.train_step()
            
            state = next_state
            total_reward += reward
            
            if terminated or truncated:
                break
        
        # Should complete without errors
        assert step > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
