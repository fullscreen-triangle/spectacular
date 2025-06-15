"""
Zengeza: Markov Decision Process Module for Spectacular

This module implements an MDP-based reasoning system that uses probabilistic 
methods, goals, and utility functions for transitioning between different states
in the visualization generation process.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import random
from collections import defaultdict, deque

# Reinforcement learning dependencies
try:
    import gym
    from gym import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    logging.warning("Gym not available. Install with: pip install gym")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisualizationState(Enum):
    """States in the visualization generation process."""
    INITIAL = "initial"
    ANALYZING_QUERY = "analyzing_query"
    SELECTING_CHART_TYPE = "selecting_chart_type"
    DESIGNING_LAYOUT = "designing_layout"
    GENERATING_CODE = "generating_code"
    ADDING_INTERACTIONS = "adding_interactions"
    OPTIMIZING_PERFORMANCE = "optimizing_performance"
    VALIDATING_OUTPUT = "validating_output"
    REFINING_RESULT = "refining_result"
    COMPLETED = "completed"
    ERROR_RECOVERY = "error_recovery"


class Action(Enum):
    """Actions available in each state."""
    ANALYZE_COMPLEXITY = "analyze_complexity"
    EXTRACT_FEATURES = "extract_features"
    SELECT_BAR_CHART = "select_bar_chart"
    SELECT_LINE_CHART = "select_line_chart"
    SELECT_SCATTER_PLOT = "select_scatter_plot"
    SELECT_HEATMAP = "select_heatmap"
    SELECT_NETWORK_DIAGRAM = "select_network_diagram"
    DESIGN_SIMPLE_LAYOUT = "design_simple_layout"
    DESIGN_COMPLEX_LAYOUT = "design_complex_layout"
    GENERATE_BASIC_CODE = "generate_basic_code"
    GENERATE_ADVANCED_CODE = "generate_advanced_code"
    ADD_TOOLTIPS = "add_tooltips"
    ADD_ZOOM = "add_zoom"
    ADD_BRUSH = "add_brush"
    OPTIMIZE_RENDERING = "optimize_rendering"
    VALIDATE_SYNTAX = "validate_syntax"
    VALIDATE_LOGIC = "validate_logic"
    REFINE_AESTHETICS = "refine_aesthetics"
    BACKTRACK = "backtrack"
    COMPLETE = "complete"
    HANDLE_ERROR = "handle_error"


@dataclass
class StateTransition:
    """Represents a state transition in the MDP."""
    from_state: VisualizationState
    action: Action
    to_state: VisualizationState
    probability: float
    reward: float
    utility: float
    timestamp: float


@dataclass
class Goal:
    """Represents a goal in the MDP."""
    id: str
    name: str
    description: str
    target_states: List[VisualizationState]
    priority: float
    completion_reward: float
    partial_rewards: Dict[VisualizationState, float] = field(default_factory=dict)


@dataclass
class UtilityComponent:
    """Component of the utility function."""
    name: str
    weight: float
    description: str
    evaluator: str  # Function name to evaluate this component


class MDPModule:
    """
    Advanced Markov Decision Process module for visualization generation.
    
    This class implements:
    1. State space modeling for visualization generation
    2. Action selection with utility maximization
    3. Probabilistic state transitions
    4. Goal-oriented planning
    5. Adaptive policy learning
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the MDP module."""
        self.config = config or {}
        self.ready = False
        
        # MDP components
        self.current_state = VisualizationState.INITIAL
        self.state_history: List[VisualizationState] = []
        self.action_history: List[Action] = []
        self.transition_model: Dict[Tuple[VisualizationState, Action], Dict[VisualizationState, float]] = {}
        self.reward_function: Dict[Tuple[VisualizationState, Action, VisualizationState], float] = {}
        self.value_function: Dict[VisualizationState, float] = {}
        self.policy: Dict[VisualizationState, Dict[Action, float]] = {}
        
        # Goals and utilities
        self.goals: Dict[str, Goal] = {}
        self.utility_components: List[UtilityComponent] = []
        self.transition_history: List[StateTransition] = []
        
        # Learning parameters
        self.learning_rate = config.get('learning_rate', 0.1)
        self.discount_factor = config.get('discount_factor', 0.95)
        self.exploration_rate = config.get('exploration_rate', 0.1)
        self.exploration_decay = config.get('exploration_decay', 0.99)
        
        # Performance tracking
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.success_rates: Dict[str, float] = {}
        
        # Initialize the MDP
        asyncio.create_task(self._initialize_mdp())
        
        logger.info("Zengeza MDP Module initialized")
    
    async def _initialize_mdp(self):
        """Initialize the MDP components."""
        try:
            # Define goals
            await self._define_goals()
            
            # Set up utility function
            await self._setup_utility_function()
            
            # Initialize transition model
            await self._initialize_transition_model()
            
            # Initialize reward function
            await self._initialize_reward_function()
            
            # Initialize value function and policy
            await self._initialize_value_function()
            await self._initialize_policy()
            
            self.ready = True
            logger.info("MDP initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing MDP: {e}")
            self.ready = False
    
    async def _define_goals(self):
        """Define goals for the visualization generation process."""
        
        goals = [
            Goal(
                id="G001",
                name="generate_correct_visualization",
                description="Generate syntactically and semantically correct visualization",
                target_states=[VisualizationState.COMPLETED],
                priority=1.0,
                completion_reward=100.0,
                partial_rewards={
                    VisualizationState.GENERATING_CODE: 20.0,
                    VisualizationState.VALIDATING_OUTPUT: 40.0
                }
            ),
            Goal(
                id="G002", 
                name="optimize_performance",
                description="Create efficient and fast-rendering visualization",
                target_states=[VisualizationState.OPTIMIZING_PERFORMANCE, VisualizationState.COMPLETED],
                priority=0.7,
                completion_reward=50.0,
                partial_rewards={
                    VisualizationState.OPTIMIZING_PERFORMANCE: 30.0
                }
            ),
            Goal(
                id="G003",
                name="minimize_generation_time",
                description="Complete visualization generation quickly",
                target_states=[VisualizationState.COMPLETED],
                priority=0.5,
                completion_reward=30.0
            ),
            Goal(
                id="G004",
                name="maximize_user_satisfaction",
                description="Create visually appealing and functional visualization",
                target_states=[VisualizationState.ADDING_INTERACTIONS, VisualizationState.COMPLETED],
                priority=0.8,
                completion_reward=70.0,
                partial_rewards={
                    VisualizationState.DESIGNING_LAYOUT: 15.0,
                    VisualizationState.ADDING_INTERACTIONS: 35.0
                }
            )
        ]
        
        for goal in goals:
            self.goals[goal.id] = goal
        
        logger.info(f"Defined {len(goals)} goals")
    
    async def _setup_utility_function(self):
        """Set up the utility function components."""
        
        components = [
            UtilityComponent(
                name="correctness",
                weight=0.4,
                description="How correct and valid the visualization is",
                evaluator="evaluate_correctness"
            ),
            UtilityComponent(
                name="efficiency",
                weight=0.25,
                description="How efficient the generation process is",
                evaluator="evaluate_efficiency"
            ),
            UtilityComponent(
                name="aesthetics",
                weight=0.2,
                description="How visually appealing the result is",
                evaluator="evaluate_aesthetics"
            ),
            UtilityComponent(
                name="functionality",
                weight=0.15,
                description="How functional and interactive the visualization is",
                evaluator="evaluate_functionality"
            )
        ]
        
        self.utility_components = components
        logger.info(f"Set up utility function with {len(components)} components")
    
    async def _initialize_transition_model(self):
        """Initialize the state transition model."""
        
        # Define transition probabilities for each state-action pair
        transitions = {
            # From INITIAL state
            (VisualizationState.INITIAL, Action.ANALYZE_COMPLEXITY): {
                VisualizationState.ANALYZING_QUERY: 0.9,
                VisualizationState.ERROR_RECOVERY: 0.1
            },
            
            # From ANALYZING_QUERY state
            (VisualizationState.ANALYZING_QUERY, Action.EXTRACT_FEATURES): {
                VisualizationState.SELECTING_CHART_TYPE: 0.8,
                VisualizationState.ANALYZING_QUERY: 0.15,  # Need more analysis
                VisualizationState.ERROR_RECOVERY: 0.05
            },
            
            # From SELECTING_CHART_TYPE state
            (VisualizationState.SELECTING_CHART_TYPE, Action.SELECT_BAR_CHART): {
                VisualizationState.DESIGNING_LAYOUT: 0.85,
                VisualizationState.SELECTING_CHART_TYPE: 0.1,  # Reconsider
                VisualizationState.ERROR_RECOVERY: 0.05
            },
            (VisualizationState.SELECTING_CHART_TYPE, Action.SELECT_LINE_CHART): {
                VisualizationState.DESIGNING_LAYOUT: 0.85,
                VisualizationState.SELECTING_CHART_TYPE: 0.1,
                VisualizationState.ERROR_RECOVERY: 0.05
            },
            (VisualizationState.SELECTING_CHART_TYPE, Action.SELECT_SCATTER_PLOT): {
                VisualizationState.DESIGNING_LAYOUT: 0.8,
                VisualizationState.SELECTING_CHART_TYPE: 0.15,
                VisualizationState.ERROR_RECOVERY: 0.05
            },
            (VisualizationState.SELECTING_CHART_TYPE, Action.SELECT_HEATMAP): {
                VisualizationState.DESIGNING_LAYOUT: 0.75,
                VisualizationState.SELECTING_CHART_TYPE: 0.2,
                VisualizationState.ERROR_RECOVERY: 0.05
            },
            
            # From DESIGNING_LAYOUT state
            (VisualizationState.DESIGNING_LAYOUT, Action.DESIGN_SIMPLE_LAYOUT): {
                VisualizationState.GENERATING_CODE: 0.9,
                VisualizationState.DESIGNING_LAYOUT: 0.05,
                VisualizationState.ERROR_RECOVERY: 0.05
            },
            (VisualizationState.DESIGNING_LAYOUT, Action.DESIGN_COMPLEX_LAYOUT): {
                VisualizationState.GENERATING_CODE: 0.7,
                VisualizationState.DESIGNING_LAYOUT: 0.2,
                VisualizationState.ERROR_RECOVERY: 0.1
            },
            
            # From GENERATING_CODE state
            (VisualizationState.GENERATING_CODE, Action.GENERATE_BASIC_CODE): {
                VisualizationState.VALIDATING_OUTPUT: 0.8,
                VisualizationState.ADDING_INTERACTIONS: 0.15,
                VisualizationState.ERROR_RECOVERY: 0.05
            },
            (VisualizationState.GENERATING_CODE, Action.GENERATE_ADVANCED_CODE): {
                VisualizationState.ADDING_INTERACTIONS: 0.6,
                VisualizationState.VALIDATING_OUTPUT: 0.25,
                VisualizationState.ERROR_RECOVERY: 0.15
            },
            
            # From ADDING_INTERACTIONS state
            (VisualizationState.ADDING_INTERACTIONS, Action.ADD_TOOLTIPS): {
                VisualizationState.ADDING_INTERACTIONS: 0.4,  # Can add more
                VisualizationState.OPTIMIZING_PERFORMANCE: 0.3,
                VisualizationState.VALIDATING_OUTPUT: 0.25,
                VisualizationState.ERROR_RECOVERY: 0.05
            },
            (VisualizationState.ADDING_INTERACTIONS, Action.ADD_ZOOM): {
                VisualizationState.ADDING_INTERACTIONS: 0.3,
                VisualizationState.OPTIMIZING_PERFORMANCE: 0.4,
                VisualizationState.VALIDATING_OUTPUT: 0.25,
                VisualizationState.ERROR_RECOVERY: 0.05
            },
            
            # From VALIDATING_OUTPUT state
            (VisualizationState.VALIDATING_OUTPUT, Action.VALIDATE_SYNTAX): {
                VisualizationState.VALIDATING_OUTPUT: 0.3,  # More validation needed
                VisualizationState.COMPLETED: 0.5,
                VisualizationState.REFINING_RESULT: 0.15,
                VisualizationState.ERROR_RECOVERY: 0.05
            },
            (VisualizationState.VALIDATING_OUTPUT, Action.VALIDATE_LOGIC): {
                VisualizationState.COMPLETED: 0.6,
                VisualizationState.REFINING_RESULT: 0.3,
                VisualizationState.ERROR_RECOVERY: 0.1
            },
            
            # From REFINING_RESULT state
            (VisualizationState.REFINING_RESULT, Action.REFINE_AESTHETICS): {
                VisualizationState.COMPLETED: 0.7,
                VisualizationState.REFINING_RESULT: 0.25,
                VisualizationState.ERROR_RECOVERY: 0.05
            },
            
            # Error recovery
            (VisualizationState.ERROR_RECOVERY, Action.BACKTRACK): {
                VisualizationState.ANALYZING_QUERY: 0.4,
                VisualizationState.SELECTING_CHART_TYPE: 0.3,
                VisualizationState.GENERATING_CODE: 0.2,
                VisualizationState.ERROR_RECOVERY: 0.1
            }
        }
        
        self.transition_model = transitions
        logger.info(f"Initialized transition model with {len(transitions)} transitions")
    
    async def _initialize_reward_function(self):
        """Initialize the reward function."""
        
        # Positive rewards for successful transitions
        rewards = {
            # Progress rewards
            (VisualizationState.INITIAL, Action.ANALYZE_COMPLEXITY, VisualizationState.ANALYZING_QUERY): 5.0,
            (VisualizationState.ANALYZING_QUERY, Action.EXTRACT_FEATURES, VisualizationState.SELECTING_CHART_TYPE): 10.0,
            (VisualizationState.SELECTING_CHART_TYPE, Action.SELECT_BAR_CHART, VisualizationState.DESIGNING_LAYOUT): 15.0,
            (VisualizationState.DESIGNING_LAYOUT, Action.DESIGN_SIMPLE_LAYOUT, VisualizationState.GENERATING_CODE): 20.0,
            (VisualizationState.GENERATING_CODE, Action.GENERATE_BASIC_CODE, VisualizationState.VALIDATING_OUTPUT): 25.0,
            (VisualizationState.VALIDATING_OUTPUT, Action.VALIDATE_SYNTAX, VisualizationState.COMPLETED): 50.0,
            
            # Quality rewards
            (VisualizationState.ADDING_INTERACTIONS, Action.ADD_TOOLTIPS, VisualizationState.OPTIMIZING_PERFORMANCE): 15.0,
            (VisualizationState.OPTIMIZING_PERFORMANCE, Action.OPTIMIZE_RENDERING, VisualizationState.VALIDATING_OUTPUT): 20.0,
            (VisualizationState.REFINING_RESULT, Action.REFINE_AESTHETICS, VisualizationState.COMPLETED): 30.0,
            
            # Penalty for errors
            (VisualizationState.GENERATING_CODE, Action.GENERATE_ADVANCED_CODE, VisualizationState.ERROR_RECOVERY): -20.0,
            (VisualizationState.VALIDATING_OUTPUT, Action.VALIDATE_LOGIC, VisualizationState.ERROR_RECOVERY): -15.0,
        }
        
        # Add default small negative reward for staying in same state (encourage progress)
        for state in VisualizationState:
            for action in Action:
                if (state, action, state) not in rewards:
                    rewards[(state, action, state)] = -1.0
        
        self.reward_function = rewards
        logger.info(f"Initialized reward function with {len(rewards)} reward mappings")
    
    async def _initialize_value_function(self):
        """Initialize the value function for each state."""
        
        # Initialize values based on proximity to goal states
        values = {
            VisualizationState.COMPLETED: 100.0,  # Terminal goal state
            VisualizationState.VALIDATING_OUTPUT: 80.0,
            VisualizationState.REFINING_RESULT: 75.0,
            VisualizationState.OPTIMIZING_PERFORMANCE: 70.0,
            VisualizationState.ADDING_INTERACTIONS: 60.0,
            VisualizationState.GENERATING_CODE: 50.0,
            VisualizationState.DESIGNING_LAYOUT: 40.0,
            VisualizationState.SELECTING_CHART_TYPE: 30.0,
            VisualizationState.ANALYZING_QUERY: 20.0,
            VisualizationState.INITIAL: 10.0,
            VisualizationState.ERROR_RECOVERY: 5.0
        }
        
        self.value_function = values
        logger.info("Initialized value function")
    
    async def _initialize_policy(self):
        """Initialize the policy (action probabilities for each state)."""
        
        policy = {}
        
        for state in VisualizationState:
            policy[state] = {}
            
            # Get valid actions for this state
            valid_actions = await self._get_valid_actions(state)
            
            if valid_actions:
                # Initialize with uniform probabilities
                prob = 1.0 / len(valid_actions)
                for action in valid_actions:
                    policy[state][action] = prob
            else:
                # If no valid actions, set empty dict
                policy[state] = {}
        
        self.policy = policy
        logger.info("Initialized policy")
    
    async def optimize_strategy(self, query_context, reasoning_results: Dict) -> Dict[str, Any]:
        """Optimize the generation strategy using MDP planning."""
        logger.info("Optimizing strategy with MDP planning...")
        
        # Reset episode
        self.current_state = VisualizationState.INITIAL
        self.state_history = [self.current_state]
        self.action_history = []
        episode_reward = 0.0
        
        # Plan action sequence
        planned_actions = await self._plan_action_sequence(query_context, reasoning_results)
        
        # Execute plan and track results
        execution_results = []
        for action in planned_actions:
            result = await self._execute_action(action)
            execution_results.append(result)
            episode_reward += result.get('reward', 0.0)
            
            # Check if we reached terminal state
            if self.current_state == VisualizationState.COMPLETED:
                break
        
        # Update policy based on episode
        await self._update_policy(episode_reward)
        
        # Record episode statistics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(len(self.action_history))
        
        return {
            "strategy": "mdp_optimized",
            "planned_actions": [a.value for a in planned_actions],
            "executed_actions": len(execution_results),
            "episode_reward": episode_reward,
            "final_state": self.current_state.value,
            "confidence": min(1.0, episode_reward / 100.0),  # Normalize reward to confidence
            "execution_results": execution_results
        }
    
    async def _plan_action_sequence(self, query_context, reasoning_results: Dict) -> List[Action]:
        """Plan a sequence of actions using value iteration or policy iteration."""
        
        # Extract context features to inform planning
        complexity = getattr(query_context, 'complexity_score', 0.5)
        visual_complexity = getattr(query_context, 'visual_complexity', 0.5)
        
        planned_actions = []
        current_state = self.current_state
        max_steps = 20  # Prevent infinite loops
        
        for step in range(max_steps):
            if current_state == VisualizationState.COMPLETED:
                break
            
            # Select action based on policy and context
            action = await self._select_action(current_state, complexity, visual_complexity)
            if not action:
                break
            
            planned_actions.append(action)
            
            # Predict next state
            next_state = await self._predict_next_state(current_state, action)
            current_state = next_state
        
        return planned_actions
    
    async def _select_action(self, state: VisualizationState, complexity: float, visual_complexity: float) -> Optional[Action]:
        """Select the best action for the current state."""
        
        valid_actions = await self._get_valid_actions(state)
        if not valid_actions:
            return None
        
        # Epsilon-greedy action selection with context adaptation
        if random.random() < self.exploration_rate:
            # Explore: random action
            return random.choice(valid_actions)
        else:
            # Exploit: select action with highest expected utility
            best_action = None
            best_utility = float('-inf')
            
            for action in valid_actions:
                utility = await self._calculate_action_utility(state, action, complexity, visual_complexity)
                if utility > best_utility:
                    best_utility = utility
                    best_action = action
            
            return best_action
    
    async def _calculate_action_utility(self, state: VisualizationState, action: Action, 
                                      complexity: float, visual_complexity: float) -> float:
        """Calculate the utility of taking an action in a state."""
        
        utility = 0.0
        
        # Base utility from value function
        if (state, action) in self.transition_model:
            for next_state, prob in self.transition_model[(state, action)].items():
                reward = self.reward_function.get((state, action, next_state), 0.0)
                next_value = self.value_function.get(next_state, 0.0)
                utility += prob * (reward + self.discount_factor * next_value)
        
        # Context-based adjustments
        if complexity > 0.7:  # High complexity
            if action in [Action.GENERATE_ADVANCED_CODE, Action.DESIGN_COMPLEX_LAYOUT]:
                utility += 10.0  # Bonus for complex actions when complexity is high
            elif action in [Action.GENERATE_BASIC_CODE, Action.DESIGN_SIMPLE_LAYOUT]:
                utility -= 5.0  # Penalty for simple actions when complexity is high
        else:  # Low complexity
            if action in [Action.GENERATE_BASIC_CODE, Action.DESIGN_SIMPLE_LAYOUT]:
                utility += 5.0  # Bonus for simple actions when complexity is low
        
        if visual_complexity > 0.6:  # High visual complexity
            if action in [Action.ADD_TOOLTIPS, Action.ADD_ZOOM, Action.ADD_BRUSH]:
                utility += 8.0  # Bonus for interaction actions
        
        return utility
    
    async def _predict_next_state(self, state: VisualizationState, action: Action) -> VisualizationState:
        """Predict the next state given current state and action."""
        
        if (state, action) not in self.transition_model:
            return state  # Stay in same state if no transition defined
        
        transitions = self.transition_model[(state, action)]
        
        # Sample next state based on probabilities
        rand = random.random()
        cumulative_prob = 0.0
        
        for next_state, prob in transitions.items():
            cumulative_prob += prob
            if rand <= cumulative_prob:
                return next_state
        
        # Fallback to most likely state
        return max(transitions.items(), key=lambda x: x[1])[0]
    
    async def _execute_action(self, action: Action) -> Dict[str, Any]:
        """Execute an action and transition to next state."""
        
        start_state = self.current_state
        next_state = await self._predict_next_state(start_state, action)
        
        # Calculate reward
        reward = self.reward_function.get((start_state, action, next_state), 0.0)
        
        # Update state
        self.current_state = next_state
        self.state_history.append(next_state)
        self.action_history.append(action)
        
        # Record transition
        transition = StateTransition(
            from_state=start_state,
            action=action,
            to_state=next_state,
            probability=self.transition_model.get((start_state, action), {}).get(next_state, 0.0),
            reward=reward,
            utility=await self._calculate_action_utility(start_state, action, 0.5, 0.5),
            timestamp=asyncio.get_event_loop().time()
        )
        self.transition_history.append(transition)
        
        return {
            "action": action.value,
            "from_state": start_state.value,
            "to_state": next_state.value,
            "reward": reward,
            "success": next_state != VisualizationState.ERROR_RECOVERY
        }
    
    async def _update_policy(self, episode_reward: float):
        """Update policy based on episode performance."""
        
        # Simple policy update: adjust action probabilities based on rewards received
        if len(self.transition_history) == 0:
            return
        
        # Calculate returns for each state-action pair in the episode
        returns = {}
        G = 0.0  # Return
        
        # Work backwards through the episode
        for i in reversed(range(len(self.transition_history))):
            transition = self.transition_history[i]
            G = transition.reward + self.discount_factor * G
            
            state_action = (transition.from_state, transition.action)
            if state_action not in returns:
                returns[state_action] = []
            returns[state_action].append(G)
        
        # Update policy using Monte Carlo method
        for (state, action), return_list in returns.items():
            avg_return = np.mean(return_list)
            
            # Update action probability based on average return
            if state in self.policy and action in self.policy[state]:
                current_prob = self.policy[state][action]
                # Simple update rule: increase probability if return is positive
                if avg_return > 0:
                    new_prob = min(1.0, current_prob + self.learning_rate * 0.1)
                else:
                    new_prob = max(0.01, current_prob - self.learning_rate * 0.1)
                
                self.policy[state][action] = new_prob
        
        # Normalize probabilities for each state
        for state in self.policy:
            total_prob = sum(self.policy[state].values())
            if total_prob > 0:
                for action in self.policy[state]:
                    self.policy[state][action] /= total_prob
        
        # Decay exploration rate
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(0.01, self.exploration_rate)
    
    async def _get_valid_actions(self, state: VisualizationState) -> List[Action]:
        """Get valid actions for a given state."""
        
        valid_actions = []
        
        if state == VisualizationState.INITIAL:
            valid_actions = [Action.ANALYZE_COMPLEXITY]
        elif state == VisualizationState.ANALYZING_QUERY:
            valid_actions = [Action.EXTRACT_FEATURES]
        elif state == VisualizationState.SELECTING_CHART_TYPE:
            valid_actions = [
                Action.SELECT_BAR_CHART,
                Action.SELECT_LINE_CHART,
                Action.SELECT_SCATTER_PLOT,
                Action.SELECT_HEATMAP,
                Action.SELECT_NETWORK_DIAGRAM
            ]
        elif state == VisualizationState.DESIGNING_LAYOUT:
            valid_actions = [Action.DESIGN_SIMPLE_LAYOUT, Action.DESIGN_COMPLEX_LAYOUT]
        elif state == VisualizationState.GENERATING_CODE:
            valid_actions = [Action.GENERATE_BASIC_CODE, Action.GENERATE_ADVANCED_CODE]
        elif state == VisualizationState.ADDING_INTERACTIONS:
            valid_actions = [Action.ADD_TOOLTIPS, Action.ADD_ZOOM, Action.ADD_BRUSH]
        elif state == VisualizationState.OPTIMIZING_PERFORMANCE:
            valid_actions = [Action.OPTIMIZE_RENDERING]
        elif state == VisualizationState.VALIDATING_OUTPUT:
            valid_actions = [Action.VALIDATE_SYNTAX, Action.VALIDATE_LOGIC]
        elif state == VisualizationState.REFINING_RESULT:
            valid_actions = [Action.REFINE_AESTHETICS]
        elif state == VisualizationState.ERROR_RECOVERY:
            valid_actions = [Action.BACKTRACK, Action.HANDLE_ERROR]
        elif state == VisualizationState.COMPLETED:
            valid_actions = [Action.COMPLETE]
        
        return valid_actions
    
    def is_ready(self) -> bool:
        """Check if the MDP module is ready."""
        return self.ready
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0.0
        
        return {
            "ready": self.ready,
            "current_state": self.current_state.value,
            "total_episodes": len(self.episode_rewards),
            "average_episode_reward": avg_reward,
            "average_episode_length": avg_length,
            "exploration_rate": self.exploration_rate,
            "total_transitions": len(self.transition_history),
            "gym_available": GYM_AVAILABLE
        }
    
    def get_policy_summary(self) -> Dict[str, Any]:
        """Get a summary of the current policy."""
        policy_summary = {}
        
        for state, action_probs in self.policy.items():
            if action_probs:
                best_action = max(action_probs.items(), key=lambda x: x[1])
                policy_summary[state.value] = {
                    "best_action": best_action[0].value,
                    "best_action_probability": best_action[1],
                    "total_actions": len(action_probs)
                }
        
        return policy_summary