import { useState } from "react";
import { sendMessage } from "./api";
import "./index.css";

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  async function handleSend() {
    if (!input.trim()) return;

    const userMsg = { role: "user", content: input };
    setMessages((prev) => [...prev, userMsg]);

    const reply = await sendMessage(input);

    const botMsg = { role: "assistant", content: reply };
    setMessages((prev) => [...prev, botMsg]);

    setInput("");
  }

  return (
    <div className="container">
      <h1>TensorFlow Agent</h1>

      <div className="chat-box">
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            {m.content}
          </div>
        ))}
      </div>

      <div className="input-row">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask the agent anything..."
        />
        <button onClick={handleSend}>Send</button>
      </div>
    </div>
  );
}
