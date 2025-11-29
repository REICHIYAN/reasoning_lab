from __future__ import annotations
from .states import State

class FSMController:
    def next_state(self, state: State, has_tool_error: bool = False) -> State:
        if state == State.ASK:
            return State.RETRIEVE
        if state == State.RETRIEVE:
            return State.REASON
        if state == State.REASON:
            return State.TOOL if has_tool_error else State.ANSWER
        if state == State.TOOL:
            return State.REASON if not has_tool_error else State.ERROR
        return State.ANSWER
