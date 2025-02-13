"""Text processing utilities for the agent."""
from typing import Any, Optional, Union, Dict, List, TypeVar, cast
import json
import re
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)
JsonType = Union[str, List[Union[str, Dict[str, Any]]]]
JsonValue = Union[str, bytes, bytearray]
StrOrBytes = Union[str, bytes]
StrOrList = Union[str, List[Union[str, Dict[str, Any]]]]

def ensure_str(value: Any) -> str:
    """Ensure value is converted to string."""
    if isinstance(value, (str, bytes, bytearray)):
        return str(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return str(value)

def safe_json_loads(text: str) -> Any:
    """Safely load JSON string."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text

class TextProcessor:
    """Text processing utilities."""
    
    def __init__(self, think_tags_pattern: str = r'<think>.*?</think>'):
        """Initialize text processor with think tags pattern."""
        self.think_tags = re.compile(think_tags_pattern, re.DOTALL)
    
    def remove_think_tags(self, text: Any) -> str:
        """Remove think tags from text.
        
        Args:
            text: Any input that can be converted to string
            
        Returns:
            String with think tags removed
        """
        try:
            text_str = ensure_str(text)
            return re.sub(self.think_tags, '', text_str)
        except Exception:
            return ""

    def process_text(self, text: Optional[Any]) -> Optional[str]:
        """Process text by removing think tags and handling different input types.
        
        Args:
            text: Any input that can be converted to string or None
            
        Returns:
            Processed string or None if input is None or processing fails
        """
        if text is None:
            return None
        try:
            text_str = ensure_str(text)
            return self.remove_think_tags(text_str)
        except Exception:
            return None

    def extract_json(self, content: Any) -> str:
        """Extract JSON from model output.
        
        Args:
            content: Any input that can be converted to string
            
        Returns:
            Valid JSON string
            
        Raises:
            ValueError: If valid JSON cannot be extracted
        """
        content_str = ensure_str(content)
        try:
            # Try to parse as pure JSON first
            safe_json_loads(content_str)
            return content_str
        except Exception:
            # If not pure JSON, try to extract JSON part
            json_match = re.search(r'\{.*\}', content_str, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                # Verify it's valid JSON
                safe_json_loads(json_str)
                return json_str
            raise ValueError(f"Could not extract valid JSON from content: {content_str}")

    def parse_json(self, text: Any) -> Any:
        """Parse JSON string into Python object.
        
        Args:
            text: Any input that can be converted to string
            
        Returns:
            Parsed JSON object or original input if parsing fails
        """
        try:
            text_str = ensure_str(text)
            return safe_json_loads(text_str)
        except Exception:
            return text

    def convert_to_json_value(self, text: Any) -> JsonValue:
        """Convert any input to a JSON-compatible value.
        
        Args:
            text: Any input
            
        Returns:
            JSON-compatible value (str, bytes, or bytearray)
        """
        try:
            if isinstance(text, (str, bytes, bytearray)):
                return cast(JsonValue, text)
            return cast(JsonValue, ensure_str(text))
        except Exception:
            return cast(JsonValue, str(text)) 