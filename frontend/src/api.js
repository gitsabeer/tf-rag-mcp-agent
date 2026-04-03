import axios from "axios";

export async function sendMessage(message) {
  const res = await axios.post("http://localhost:8001/chat", { message });
  return res.data.response;
}
