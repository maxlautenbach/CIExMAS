from typing import Literal
import traceback
from langgraph.types import Command
from approaches.full_sentence.One_Agent.setup import cIEState

def agent(state: cIEState) -> Command[Literal] | tuple[cIEState, str]:
    # Extract the indices of messages to delete from the instruction
    instruction = state["instruction"]
    try:
        # Parse message indices from the instruction - expecting comma-separated numbers
        indices_to_delete = [int(idx.strip()) for idx in instruction.split(',')]
        
        # Sort indices in descending order to avoid index shifting when deleting
        indices_to_delete.sort(reverse=True)
        
        # Check if indices are valid
        if any(idx < 0 or idx >= len(state["messages"]) for idx in indices_to_delete):
            response = "SYSTEM MESSAGE: Some of the specified message indices are out of range. No messages were deleted."
            
            if state["debug"]:
                state["instruction"] = ""
                state["messages"].append(response)
                return state, response
                
            return Command(goto="main_agent", update={"messages": state["messages"] + [response], "instruction": ""})
        
        # Create a copy of the messages list to modify
        messages = state["messages"].copy()
        
        # Store deleted message content for reference in the response
        deleted_messages = []
        for idx in indices_to_delete:
            if 0 <= idx < len(messages):
                deleted_messages.append((idx, messages[idx][:100] + "..." if len(messages[idx]) > 100 else messages[idx]))
        
        # Delete messages at the specified indices
        for idx in indices_to_delete:
            del messages[idx]
        
        # Format the response with details about what was deleted
        response = f"Message Deletion Results:\n"
        response += f"Deleted {len(indices_to_delete)} message(s) from history.\n\n"
        
        if deleted_messages:
            response += "Summary of deleted messages:\n"
            for idx, content_preview in deleted_messages:
                response += f"  Message {idx}: {content_preview}\n"
        
        if state["debug"]:
            state["messages"] = messages
            state["messages"].append(response)
            state["instruction"] = ""
            return state, response
        
        # Return command to go back to main_agent with updated messages
        return Command(goto="main_agent", update={"messages": messages + [response], "instruction": ""})
        
    except ValueError as e:
        # Handle the case where the instruction cannot be parsed correctly
        error_message = f"SYSTEM MESSAGE: Invalid format for delete_message_tool. Expected comma-separated message indices (e.g., '2,5,7'). Error: {str(e)}"
        
        if state["debug"]:
            state["instruction"] = error_message
            return state, f"Error in delete_message_tool: {str(e)}"
            
        return Command(goto="main_agent", update={"messages": state["messages"] + [error_message], "instruction": ""})
    except Exception as e:
        # Handle any other exceptions
        error_message = f"SYSTEM MESSAGE: Error occurred in delete_message_tool: {str(e)}\n{traceback.format_exc()}"
        
        if state["debug"]:
            state["instruction"] = error_message
            return state, f"Error in delete_message_tool: {str(e)}"
            
        return Command(goto="main_agent", update={"messages": state["messages"] + [error_message], "instruction": ""})