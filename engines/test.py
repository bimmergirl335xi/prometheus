import sys
import io
import inspect
import types
import copy
import pickle
import traceback
import ast
import builtins
from typing import Any, Dict, List, Optional, Callable, Union
from collections import defaultdict
from contextlib import redirect_stdout, redirect_stderr

from config import *
from utils import log, Term, log_engine_event
from definitions import Action, ActionType

# ------------------------------------------------------------------------------
# 1. THE SAFETY NET (Backend Support)
# ------------------------------------------------------------------------------

class StateSnapshot:
    """
    Captures a frozen state of an object or engine.
    Used for rolling back if the agent breaks itself.
    """
    def __init__(self, target_obj: Any):
        self.timestamp = __import__('time').time()
        try:
            # Deepcopy is risky for complex objects with locks/threads.
            # We use pickle as a safer intermediate for pure data/logic states.
            self.state_data = pickle.dumps(target_obj)
            self.success = True
        except Exception as e:
            self.success = False
            self.error = str(e)

    def restore(self, target_obj: Any):
        if not self.success:
            raise RuntimeError(f"Snapshot failed: {self.error}")
        
        previous_state = pickle.loads(self.state_data)
        # Restore attributes
        target_obj.__dict__.update(previous_state.__dict__)

class SandboxSecurity(ast.NodeTransformer):
    """
    AST Walker to prevent the agent from doing catastrophic OS damage.
    (e.g., bans 'import os', 'subprocess', 'eval')
    """
    BANNED_MODULES = {'os', 'sys', 'subprocess', 'shutil'}
    BANNED_FUNCTIONS = {'eval', 'exec', 'open'}

    def visit_Import(self, node):
        for name in node.names:
            if name.name in self.BANNED_MODULES:
                raise SecurityError(f"Importing '{name.name}' is forbidden by Safety Protocols.")
        return node

    def visit_ImportFrom(self, node):
        if node.module in self.BANNED_MODULES:
            raise SecurityError(f"Importing from '{node.module}' is forbidden.")
        return node

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id in self.BANNED_FUNCTIONS:
                raise SecurityError(f"Function '{node.func.id}' is restricted.")
        return self.generic_visit(node)

class SecurityError(Exception):
    pass

# ------------------------------------------------------------------------------
# 2. THE INFINITY ENGINE (The Meta-Controller)
# ------------------------------------------------------------------------------

class InfinityEngine:
    """
    The 'God Mode' engine for the agent.
    Allows dynamic wiring, code execution, and self-modification.
    """
    def __init__(self, agent_ref: Any):
        self.agent = agent_ref  # Reference to the main Agent class
        self.registry: Dict[str, Any] = {} # Map of "ReasoningEngine" -> instance
        self.synapses: Dict[str, List[Callable]] = defaultdict(list)
        self.snapshots: Dict[str, StateSnapshot] = {}
        
        # Define the execution environment (The Sandbox)
        self.execution_context = {
            "agent": self.agent,
            "log": log,
            "Term": Term,
            "np": __import__('numpy'),
            "Action": Action,
            "ActionType": ActionType
        }

    def register_engine(self, name: str, instance: Any):
        """Register a cognitive engine to be controllable."""
        self.registry[name] = instance
        self.execution_context[name] = instance
        log("INFINITY", f"🔗 Neural Link Established: {name}", Term.GREEN)

    # --- CAPABILITY A: DYNAMIC WIRING (Synapses) ---

    def create_synapse(self, source_engine: str, event_trigger: str, target_code: str):
        """
        Dynamically links two engines. 
        When 'source_engine' triggers 'event_trigger', 'target_code' is executed.
        """
        synapse_id = f"{source_engine}:{event_trigger}->{hash(target_code)}"
        
        def synapse_callback(data):
            log("INFINITY", f"⚡ Synapse Fired: {synapse_id}", Term.PURPLE)
            # Inject the event data into the scope
            local_scope = {"event_data": data}
            self.execute_script(target_code, extra_context=local_scope)

        # Store the hook (In a real implementation, engines would need an event bus)
        # For now, we simulate it by monkey-patching the source engine's output method
        self.synapses[synapse_id].append(synapse_callback)
        self._inject_hook(source_engine, event_trigger, synapse_callback)
        return synapse_id

    def _inject_hook(self, engine_name: str, method_name: str, callback: Callable):
        """
        Monkey-patches a method on an engine to fire a callback when run.
        """
        engine = self.registry.get(engine_name)
        if not engine: return
        
        original_method = getattr(engine, method_name, None)
        if not original_method: return

        def hooked_method(*args, **kwargs):
            # Run original
            result = original_method(*args, **kwargs)
            # Fire callback with result
            try:
                callback(result)
            except Exception as e:
                log("INFINITY", f"⚠️ Synapse Failure: {e}", Term.RED)
            return result

        setattr(engine, method_name, hooked_method)
        log("INFINITY", f"💉 Injected Hook into {engine_name}.{method_name}", Term.YELLOW)

    # --- CAPABILITY B: HOT-SWAPPING (Self-Modification) ---

    def patch_method(self, target_engine: str, method_name: str, new_code_str: str):
        """
        Completely rewrites a method of a running engine class at runtime.
        Includes automatic rollback on failure.
        """
        engine = self.registry.get(target_engine)
        if not engine:
            return False, "Engine not found."

        # 1. Safety Snapshot
        log("INFINITY", f"🛡️ Creating snapshot of {target_engine} before patching...", Term.CYAN)
        snapshot = StateSnapshot(engine)
        if not snapshot.success:
            return False, f"Snapshot failed: {snapshot.error}"

        # 2. Prepare Code
        # We wrap the user's code in a function definition
        full_code = f"def {method_name}(self, *args, **kwargs):\n"
        for line in new_code_str.split('\n'):
            full_code += f"    {line}\n"

        try:
            # 3. Compile & Bind
            temp_scope = {}
            exec(full_code, self.execution_context, temp_scope)
            new_func = temp_scope[method_name]
            
            # Bind to instance
            bound_method = types.MethodType(new_func, engine)
            setattr(engine, method_name, bound_method)
            
            log("INFINITY", f"🧬 Successfully patched {target_engine}.{method_name}", Term.GREEN)
            return True, "Patch applied."

        except Exception as e:
            # 4. Rollback
            log("INFINITY", f"🔥 Patch crashed! Rolling back {target_engine}...", Term.RED)
            try:
                snapshot.restore(engine)
                log("INFINITY", f"✅ Rollback successful.", Term.GREEN)
            except Exception as rb_e:
                log("INFINITY", f"💀 CRITICAL: Rollback failed! {rb_e}", Term.RED)
            
            return False, str(e)

    # --- CAPABILITY C: SANDBOXED EXECUTION ---

    def execute_script(self, script: str, extra_context: Dict = None) -> Any:
        """
        Executes arbitrary Python code generated by the agent.
        """
        # 1. Security Check
        try:
            tree = ast.parse(script)
            SandboxSecurity().visit(tree)
        except SecurityError as e:
            log("INFINITY", f"🚫 Security Block: {e}", Term.RED)
            return None
        except SyntaxError as e:
            log("INFINITY", f"❌ Syntax Error: {e}", Term.RED)
            return None

        # 2. Context Merging
        local_scope = self.execution_context.copy()
        if extra_context:
            local_scope.update(extra_context)
            
        # 3. Execution with Capture
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        result = None
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # We use exec() but we try to extract the last expression if possible
                # (Like a REPL)
                exec(script, local_scope)
                
                # Try to find a 'result' variable if the script set one
                if 'result' in local_scope:
                    result = local_scope['result']

            output = stdout_capture.getvalue()
            if output.strip():
                log("INFINITY_OUT", output, Term.WHITE)
                
            return result

        except Exception as e:
            err = traceback.format_exc()
            log("INFINITY", f"💥 Runtime Error:\n{err}", Term.RED)
            return None

    # --- INTEGRATION WITH ACTION SYSTEM ---

    def process_request(self, action: Action):
        """
        Handles ActionType.MODIFY_SELF or ActionType.EXPERIMENTAL_CODE
        """
        payload = action.payload
        if not isinstance(payload, dict): return

        intent = payload.get("intent", "")
        
        if intent == "execute_code":
            code = payload.get("code", "")
            self.execute_script(code)
            
        elif intent == "patch_method":
            target = payload.get("target_engine")
            method = payload.get("method")
            code = payload.get("code")
            self.patch_method(target, method, code)
            
        elif intent == "wire_synapse":
            self.create_synapse(
                payload.get("source"),
                payload.get("event"),
                payload.get("target_code")
            )