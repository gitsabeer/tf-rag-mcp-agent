import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { ChatService } from './services/chat.service';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';


interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet,FormsModule,CommonModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  title = 'TensorFlow Agent (Angular)';
  messages: ChatMessage[] = [];
  input = '';
  loading = false;

  constructor(private chatService: ChatService) {}

  send() {
    const text = this.input.trim();
    if (!text || this.loading) return;

    const userMsg: ChatMessage = { role: 'user', content: text };
    this.messages = [...this.messages, userMsg];
    this.input = '';
    this.loading = true;

    this.chatService.sendMessage(text).subscribe({
      next: (res) => {
        const agentMsg: ChatMessage = {
          role: 'assistant',
          content: res.reply
        };
        this.messages = [...this.messages, agentMsg];
        this.loading = false;
      },
      error: (err) => {
        const errMsg: ChatMessage = {
          role: 'assistant',
          content: 'Error talking to backend: ' + (err?.message || 'Unknown error')
        };
        this.messages = [...this.messages, errMsg];
        this.loading = false;
      }
    });
  }

  onKeydown(event: KeyboardEvent) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.send();
    }
  }
}