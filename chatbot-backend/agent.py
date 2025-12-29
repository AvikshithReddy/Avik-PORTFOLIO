"""
Agent-Based Reasoning Loop
Implements LLM + tools + memory + reasoning for autonomous responses
"""
from typing import List, Dict, Optional, Callable, Any
from enum import Enum
import json
from datetime import datetime


class AgentAction(Enum):
    """Types of actions an agent can take"""
    RETRIEVE_FROM_KB = "retrieve_from_kb"
    SEARCH_GITHUB = "search_github"
    SEARCH_RESUME = "search_resume"
    THINK = "think"
    RESPOND = "respond"
    ASK_CLARIFICATION = "ask_clarification"


class AgentTool:
    """Represents a tool available to the agent"""
    
    def __init__(self,
                 name: str,
                 description: str,
                 func: Callable,
                 required_params: List[str] = None,
                 return_type: str = "string"):
        """
        Args:
            name: Tool name
            description: Human-readable description
            func: Python callable that implements the tool
            required_params: List of required parameter names
            return_type: Expected return type
        """
        self.name = name
        self.description = description
        self.func = func
        self.required_params = required_params or []
        self.return_type = return_type
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for LLM context"""
        return {
            "name": self.name,
            "description": self.description,
            "required_params": self.required_params,
            "return_type": self.return_type,
        }
    
    async def execute(self, **kwargs) -> Any:
        """Execute the tool"""
        try:
            return self.func(**kwargs)
        except Exception as e:
            return f"Error executing {self.name}: {str(e)}"


class ReasoningStep:
    """Single step in reasoning process"""
    
    def __init__(self,
                 step_number: int,
                 thought: str,
                 action: AgentAction,
                 tool_name: Optional[str] = None,
                 observation: Optional[str] = None):
        """
        Args:
            step_number: Step number in sequence
            thought: What the agent is thinking
            action: What action to take
            tool_name: Which tool to use (if applicable)
            observation: Result from action
        """
        self.step_number = step_number
        self.thought = thought
        self.action = action
        self.tool_name = tool_name
        self.observation = observation
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            "step": self.step_number,
            "thought": self.thought,
            "action": self.action.value,
            "tool": self.tool_name,
            "observation": self.observation,
            "timestamp": self.timestamp.isoformat(),
        }


class PortfolioAgent:
    """
    Autonomous agent for answering portfolio questions
    Uses: LLM + tools + memory + reasoning loop
    """
    
    def __init__(self,
                 openai_client,
                 context_manager,
                 model: str = "gpt-3.5-turbo",
                 max_reasoning_steps: int = 5):
        """
        Args:
            openai_client: OpenAI client
            context_manager: ContextManager instance
            model: LLM model to use
            max_reasoning_steps: Max reasoning steps before responding
        """
        self.openai_client = openai_client
        self.context_manager = context_manager
        self.model = model
        self.max_reasoning_steps = max_reasoning_steps
        self.tools: Dict[str, AgentTool] = {}
        self.reasoning_history: List[ReasoningStep] = []
        self.system_prompt = self._build_system_prompt()
    
    def register_tool(self, tool: AgentTool) -> None:
        """Register a tool for the agent to use"""
        self.tools[tool.name] = tool
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with tool descriptions"""
        tools_desc = "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in self.tools.values()
        ])
        
        return f"""
You are an intelligent portfolio assistant AI agent. Your role is to answer questions about 
Avikshith Reddy's professional background, skills, projects, and experience.

You have access to these tools:
{tools_desc}

Use the ReAct (Reasoning + Acting) framework:
1. THOUGHT: Analyze what you know and what you need to find out
2. ACTION: Decide which tool to use (or respond directly)
3. OBSERVATION: Process the result

Be concise, informative, and grounded in the provided data.
Always cite sources when available.
Be honest if information is not available.
"""
    
    def _parse_action_from_response(self, response_text: str) -> tuple:
        """Parse action and parameters from LLM response"""
        # Look for patterns like "Use Tool: tool_name with param: value"
        lines = response_text.split('\n')
        
        action = AgentAction.RESPOND  # Default
        tool_name = None
        params = {}
        
        for line in lines:
            line = line.lower()
            if "tool:" in line or "use:" in line:
                # Extract tool name
                for tool in self.tools.keys():
                    if tool.lower() in line:
                        tool_name = tool
                        action = AgentAction.RETRIEVE_FROM_KB  # Default tool action
                        break
            elif "think:" in line or "thought:" in line:
                action = AgentAction.THINK
            elif "respond:" in line or "final answer:" in line:
                action = AgentAction.RESPOND
        
        return action, tool_name, params
    
    async def reason_and_respond(self,
                                 user_query: str,
                                 include_reasoning: bool = False) -> Dict:
        """
        Main reasoning loop: Think -> Act -> Observe -> Respond
        
        Args:
            user_query: User's question
            include_reasoning: Whether to include reasoning in response
            
        Returns:
            Dict with response, reasoning_steps, sources, etc.
        """
        self.reasoning_history = []
        
        # Initialize context
        context = self.context_manager.get_full_context()
        
        # Reasoning loop
        reasoning_steps = []
        current_observation = None
        
        for step_num in range(1, self.max_reasoning_steps + 1):
            # THOUGHT: Plan next action
            thought_response = await self._llm_think(
                user_query,
                context,
                current_observation,
                step_num
            )
            
            reasoning_step = ReasoningStep(
                step_number=step_num,
                thought=thought_response.get("thought", ""),
                action=AgentAction.THINK
            )
            reasoning_steps.append(reasoning_step)
            
            # ACT: Execute tool or respond
            action, tool_name, params = self._parse_action_from_response(thought_response.get("thought", ""))
            
            if action == AgentAction.RESPOND:
                # Time to respond
                final_response = await self._llm_respond(
                    user_query,
                    context,
                    reasoning_steps
                )
                
                return {
                    "response": final_response["response"],
                    "sources": final_response.get("sources", []),
                    "confidence": final_response.get("confidence", 0.8),
                    "reasoning": [s.to_dict() for s in reasoning_steps] if include_reasoning else None,
                    "steps": len(reasoning_steps),
                }
            
            # OBSERVE: Execute tool
            if tool_name and tool_name in self.tools:
                tool = self.tools[tool_name]
                observation = await tool.execute(**params)
                current_observation = observation
                
                reasoning_step = ReasoningStep(
                    step_number=step_num,
                    thought=thought_response.get("thought", ""),
                    action=AgentAction.RETRIEVE_FROM_KB,
                    tool_name=tool_name,
                    observation=str(observation)[:200]  # Limit observation length
                )
                reasoning_steps[-1] = reasoning_step
            
            context += f"\n[Step {step_num}] {thought_response.get('thought', '')}\n"
        
        # If we ran out of steps, respond anyway
        final_response = await self._llm_respond(user_query, context, reasoning_steps)
        return {
            "response": final_response["response"],
            "sources": final_response.get("sources", []),
            "confidence": final_response.get("confidence", 0.6),
            "reasoning": [s.to_dict() for s in reasoning_steps] if include_reasoning else None,
            "steps": len(reasoning_steps),
        }
    
    async def _llm_think(self,
                         user_query: str,
                         context: str,
                         observation: Optional[str],
                         step_num: int) -> Dict:
        """LLM thinking step"""
        prompt = f"""
Current Step: {step_num}/{self.max_reasoning_steps}
User Question: {user_query}
Previous Observation: {observation or "None"}

Context:
{context[:1000]}  # Limit context to avoid token limits

Based on the above, what should we do next?
- If you need more information, use a tool
- If you have enough information, prepare the final response
- Be concise in your thought process
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            return {
                "thought": response.choices[0].message.content,
                "step": step_num
            }
        except Exception as e:
            return {"thought": f"Thinking about: {user_query}", "step": step_num, "error": str(e)}
    
    async def _llm_respond(self,
                           user_query: str,
                           context: str,
                           reasoning_steps: List[ReasoningStep]) -> Dict:
        """LLM response generation"""
        reasoning_text = "\n".join([
            f"Step {s.step_number}: {s.thought}"
            for s in reasoning_steps
        ])
        
        prompt = f"""
Based on your reasoning and the provided context, give a helpful response to:
"{user_query}"

Reasoning process:
{reasoning_text}

Provide a clear, concise response with relevant information.
If citing specific skills or projects, include them.
Rate your confidence in the response (0-1).

Format your response as:
RESPONSE: [your response]
SOURCES: [list of source types used, e.g., resume, github]
CONFIDENCE: [0.0-1.0]
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            text = response.choices[0].message.content
            
            # Parse response
            response_text = ""
            sources = []
            confidence = 0.8
            
            for line in text.split('\n'):
                if line.startswith("RESPONSE:"):
                    response_text = line.replace("RESPONSE:", "").strip()
                elif line.startswith("SOURCES:"):
                    sources_str = line.replace("SOURCES:", "").strip()
                    sources = [s.strip() for s in sources_str.split(',')]
                elif line.startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line.replace("CONFIDENCE:", "").strip())
                    except:
                        confidence = 0.8
            
            return {
                "response": response_text or text,
                "sources": sources,
                "confidence": confidence
            }
        except Exception as e:
            return {
                "response": f"I encountered an error: {str(e)}",
                "sources": [],
                "confidence": 0.0
            }
