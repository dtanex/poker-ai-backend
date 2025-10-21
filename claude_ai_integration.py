"""
Claude AI Integration for Enhanced Poker Coaching
This creates infrastructure for GTO solver → Claude AI → User coaching flow
"""
import anthropic
import json
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ClaudePokerCoach:
    """
    Integration layer between GTO solver and Claude AI

    Flow:
    1. GTO Solver provides raw strategy data
    2. This class enriches it with context
    3. Claude AI generates human-readable coaching
    4. Response includes actionable insights
    """

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"

    def generate_coaching(
        self,
        hand_analysis: Dict,
        gto_strategy: Dict,
        user_action: Optional[str] = None,
        user_history: Optional[List[Dict]] = None
    ) -> str:
        """
        Generate coaching using Claude AI with full context

        Args:
            hand_analysis: Hand classification, properties, strength
            gto_strategy: GTO recommendations from solver
            user_action: What the user actually did (if reviewing)
            user_history: Recent hands for pattern detection

        Returns:
            Detailed coaching text
        """

        # Build context for Claude
        context = self._build_coaching_context(
            hand_analysis,
            gto_strategy,
            user_action,
            user_history
        )

        # Create coaching prompt
        system_prompt = self._create_system_prompt()
        user_prompt = self._create_user_prompt(context)

        # Call Claude API
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": user_prompt
                }]
            )

            coaching_text = response.content[0].text
            return coaching_text

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return self._fallback_coaching(hand_analysis, gto_strategy)

    def _build_coaching_context(
        self,
        hand_analysis: Dict,
        gto_strategy: Dict,
        user_action: Optional[str],
        user_history: Optional[List[Dict]]
    ) -> Dict:
        """Build rich context for Claude"""

        context = {
            # Hand details
            "hand": {
                "cards": hand_analysis.get("cards", []),
                "category": hand_analysis.get("category", ""),
                "strength_percentile": hand_analysis.get("properties", {}).get("strength_percentile", 0),
                "playability": hand_analysis.get("properties", {}).get("playability_score", 0),
                "blocker_value": hand_analysis.get("properties", {}).get("blocker_value", 0),
            },

            # Situation
            "situation": {
                "position": hand_analysis.get("position", ""),
                "action_facing": hand_analysis.get("action_facing", ""),
                "stack_depth": hand_analysis.get("stackSize", 100),
                "pot_size": gto_strategy.get("pot_size", 0),
            },

            # GTO recommendation
            "gto": {
                "recommended_action": gto_strategy.get("recommended_action", ""),
                "frequencies": gto_strategy.get("strategy", {}),
                "reasoning": gto_strategy.get("reasoning", ""),
            },

            # User's decision (if reviewing)
            "user_decision": {
                "action_taken": user_action,
                "correct": user_action == gto_strategy.get("recommended_action") if user_action else None,
            },

            # Historical patterns (for personalized coaching)
            "patterns": self._extract_patterns(user_history) if user_history else {},
        }

        return context

    def _create_system_prompt(self) -> str:
        """System prompt for Claude AI coaching"""
        return """You are an expert poker coach providing GTO (Game Theory Optimal) analysis and personalized feedback.

Your coaching should:
1. Explain WHY the GTO strategy makes sense (don't just state it)
2. Identify key factors: position, hand properties, stack depth, pot odds
3. Provide actionable insights the player can use in future hands
4. If reviewing a mistake, be constructive and explain the leak
5. Use poker terminology appropriately but explain complex concepts
6. Focus on teaching principles, not just memorizing charts

Format your response with clear sections:
## Hand Analysis
## GTO Strategy Explanation
## Key Takeaways
## (If applicable) Your Decision Review"""

    def _create_user_prompt(self, context: Dict) -> str:
        """Create user prompt with full context"""

        hand = context['hand']
        situation = context['situation']
        gto = context['gto']
        user_decision = context['user_decision']

        prompt = f"""Analyze this poker hand and provide coaching:

**Hand:** {' '.join(hand['cards'])}
**Position:** {situation['position']}
**Situation:** {situation['action_facing']}
**Stack Depth:** {situation['stack_depth']}BB

**Hand Properties:**
- Category: {hand['category']}
- Strength Percentile: {hand['strength_percentile']:.1f}%
- Playability Score: {hand['playability']:.1f}/100
- Blocker Value: {hand['blocker_value']:.1f}

**GTO Recommendation:**
- Action: {gto['recommended_action'].upper()}
- Frequencies: Fold {gto['frequencies'].get('fold', 0):.1%}, Call {gto['frequencies'].get('call', 0):.1%}, Raise {gto['frequencies'].get('raise', 0):.1%}
"""

        # Add user decision if reviewing
        if user_decision['action_taken']:
            if user_decision['correct']:
                prompt += f"\n**Your Decision:** {user_decision['action_taken'].upper()} ✅ CORRECT"
            else:
                prompt += f"\n**Your Decision:** {user_decision['action_taken'].upper()} ❌ (GTO recommends {gto['recommended_action'].upper()})"

        # Add patterns if available
        if context['patterns']:
            prompt += f"\n\n**Your Tendencies:**\n"
            for pattern, count in context['patterns'].items():
                prompt += f"- {pattern}: {count}\n"

        prompt += "\n\nProvide detailed coaching on this spot."

        return prompt

    def _extract_patterns(self, user_history: List[Dict]) -> Dict[str, int]:
        """Extract patterns from user's hand history"""
        patterns = {}

        if not user_history:
            return patterns

        # Count common mistakes
        overfold_count = sum(1 for h in user_history if h.get('leak') == 'overfolding')
        overaggressive_count = sum(1 for h in user_history if h.get('leak') == 'over_aggressive')
        undercall_count = sum(1 for h in user_history if h.get('leak') == 'undercalling')

        if overfold_count > 3:
            patterns['Overfolding tendency'] = overfold_count
        if overaggressive_count > 3:
            patterns['Over-aggressive tendency'] = overaggressive_count
        if undercall_count > 3:
            patterns['Not calling enough'] = undercall_count

        return patterns

    def _fallback_coaching(self, hand_analysis: Dict, gto_strategy: Dict) -> str:
        """Fallback coaching if Claude API fails"""
        action = gto_strategy.get('recommended_action', 'fold')
        frequencies = gto_strategy.get('strategy', {})

        return f"""## GTO Analysis

**Recommended Action:** {action.upper()}

**Strategy Frequencies:**
- Fold: {frequencies.get('fold', 0):.1%}
- Call: {frequencies.get('call', 0):.1%}
- Raise: {frequencies.get('raise', 0):.1%}

**Hand:** {' '.join(hand_analysis.get('cards', []))}
**Position:** {hand_analysis.get('position', '')}
**Situation:** {hand_analysis.get('action_facing', '')}

This is a simplified analysis. For detailed coaching, please try again."""


# Global instance (initialized by backend)
_coach_instance: Optional[ClaudePokerCoach] = None


def initialize_claude_coach(api_key: str):
    """Initialize global Claude coach instance"""
    global _coach_instance
    _coach_instance = ClaudePokerCoach(api_key)
    logger.info("✅ Claude AI poker coach initialized")


def get_coach() -> Optional[ClaudePokerCoach]:
    """Get global coach instance"""
    return _coach_instance
