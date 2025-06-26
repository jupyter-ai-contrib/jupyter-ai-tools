"""Tools that provide code execution features"""


import asyncio
import shlex
from typing import Optional

from jupyter_ai.tools.models import Tool, Toolkit


async def bash(command: str, timeout: Optional[int] = None) -> str:
    """Executes a bash command and returns the result
    
    Args:
        command: The bash command to execute
        timeout: Optional timeout in seconds
        
    Returns:
        The command output (stdout and stderr combined)
    """
    
    proc = await asyncio.create_subprocess_exec(
        *shlex.split(command),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout)
        output = stdout.decode("utf-8")
        error = stderr.decode("utf-8")
        
        if proc.returncode != 0:
            if error:
                return f"Error: {error}"
            return f"Command failed with exit code {proc.returncode}"
            
        return output if output else "Command executed successfully with no output."
    except asyncio.TimeoutError:
        proc.kill()
        return f"Command timed out after {timeout} seconds"
    

toolkit = Toolkit(
    name="code_execution_toolkit",
    description="Tools to execute code in different environments.",
)
toolkit.add_tool(Tool(callable=bash, execute=True))

