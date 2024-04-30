document.getElementById('sendButton').addEventListener('click', sendMessage);
document.getElementById('messageInput').addEventListener('keyup', function(event) {
    if (event.keyCode === 13) {
        sendMessage();
    }
});

function sendMessage() {
    var messageInput = document.getElementById('messageInput');
    var message = messageInput.value.trim();
    if (message !== '') {
        appendMessage('user', message);
        messageInput.value = '';
        getResponse(message);
    }
}

function appendMessage(sender, message) {
    var chatBody = document.getElementById('chatBody');
    var messageElement = document.createElement('div');
    messageElement.classList.add('message');
    if (sender === 'user') {
        messageElement.classList.add('user-message');
        messageElement.innerHTML = `<div class="message-text">${message}</div>`;
    } else {
        messageElement.classList.add('bot-message');
        messageElement.innerHTML = `<img src="/static/images/bot.png" alt="Bot Avatar" class="avatar"><div class="message-text">${message}</div>`;
    }
    chatBody.appendChild(messageElement);
    chatBody.scrollTop = chatBody.scrollHeight;
}

function getResponse(message) {
    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: `message=${encodeURIComponent(message)}`
    })
    .then(response => response.json())
    .then(data => {
        appendMessage('bot', data.response);
    })
    .catch(error => {
        console.error('Error:', error);
    });
}
