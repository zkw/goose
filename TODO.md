你这双眼睛太敏锐了，简直是灵魂拷问！完全说到点子上了。

### 原来的“黑科技”（留门机制）是怎么起作用的？

在重构前，原来的代码里有这么一句不起眼的判断：
```rust
ctx.last_msg_has_tool_call = msg.content.iter().any(|c| matches!(c, MessageContent::ToolRequest(_)));
// ...
if cur.is_none() { 
    // 如果自然结束（而不是因为工具调用中断），它会直接跳过退出逻辑
    // ...
}
```
**它的黑魔法在于**：当大模型完成一轮对话（例如说了一句“我已经派发了任务，我等着”），这轮流的最末尾是一条纯**文字 (Text)** 消息，不包含工具请求。因此 `last_msg_has_tool_call` 是 `false`。
流结束后，代码**故意没有 break 退出循环**，而是进入了死循环 `continue`，死死卡在 `rx.recv()` 等待后台事件上。
从前端 UI 的视角看：这个 SSE 流（Stream）始终没断开，相当于大模型在“屏息凝神”地等待，一旦后台有汇报，立即通过这个未断开的流发给前端。这就是所谓的“UI 挂起（留门）”。

### 重构后为什么门被关死了？

在重构时，我为了修复状态管理，写出了下面这个“一失足成千古恨”的逻辑：
```rust
let mut has_tool_call = false; // 放在了流的外部

if let AgentEvent::Message(msg) = &event {
    if msg.has_tool_request() {
        has_tool_call = true; // 只要流里出现过一次工具调用，它就永远是 true！
    }
}

if current.is_none() {
    if has_tool_call {
        break; // 强制关门！
    }
    // ... 留门逻辑
}
```
**问题就出在状态的“黏性”上**：
你要派发工程师，必然会在流的中途调用 `delegate` 工具对吧？
只要调用过一次，`has_tool_call` 就变成了 `true`。等大模型最终说完话、结束回合时，代码一看 `has_tool_call == true`，直接毫不留情地执行了 `break`。
流断开了，前端自然就收不到任何消息了，只能等用户手动发文字去戳它。

### 修复方案：恢复精准的“最后消息判定”

我们只需要将判断从“这一轮是否调用过工具”恢复为“**最后一句话是不是工具请求**”。
如果流的最后一句话是工具请求，说明流是异常中断的（比如碰到需要用户确认的高危操作），此时必须 break 把控制权交还给 UI；如果最后一句是正常的结束语（文字），并且后台还有任务，那就果断“留门”。

请应用这段补丁，你的“挂起等汇报”体验就会完美回归：

```diff
--- a/crates/goose/src/agents/platform_extensions/better_summon/middleware.rs
+++ b/crates/goose/src/agents/platform_extensions/better_summon/middleware.rs
@@ -65,7 +65,7 @@
         let mut current: Option<BoxStream<'a, Result<AgentEvent>>> = Some(inner);
         let mut ui_buffer = VecDeque::new();
         let mut is_safe = true;
-        let mut has_tool_call = false;
+        let mut is_last_msg_tool_request = false;
 
         let pipeline = stream! {
@@ -95,9 +95,7 @@
                                 let event = Self::normalize_agent_event(event, is_sub);
                                 if let AgentEvent::Message(msg) = &event {
                                     is_safe = Self::is_safe_phase(msg);
-                                    if msg.has_tool_request() {
-                                        has_tool_call = true;
-                                    }
+                                    is_last_msg_tool_request = msg.has_tool_request();
                                 }
                                 if is_safe {
                                     while let Some(buffered) = ui_buffer.pop_front() {
@@ -148,7 +146,7 @@
                         yield Ok(AgentEvent::Message(buffered));
                     }
 
-                    if has_tool_call {
+                    if is_last_msg_tool_request {
                         break;
                     }
```