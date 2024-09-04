// script.js

function askQuestion() {
    const fileInput = document.getElementById('fileInput');
    const questionInput = document.getElementById('questionInput');
    const responseTextarea = document.getElementById('responseTextarea');

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('question', questionInput.value);

    responseTextarea.value = '';  // Clear previous response
    console.log("Sending request to server...");

    // Upload the file first
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        console.log("File uploaded successfully. Proceeding to generate response...");

        // Then generate the response
        return fetch('/generate', {
            method: 'POST',
            body: new URLSearchParams({ 'question': questionInput.value })
        });
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        console.log("Received response from server, starting to read...");
        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        return new ReadableStream({
            start(controller) {
                function push() {
                    reader.read().then(({ done, value }) => {
                        if (done) {
                            console.log("Stream complete");
                            controller.close();
                            return;
                        }
                        const chunk = decoder.decode(value, { stream: true });
                        console.log("Received chunk:", chunk);
                        controller.enqueue(chunk);
                        responseTextarea.value += chunk;
                        responseTextarea.scrollTop = responseTextarea.scrollHeight;  // Auto-scroll to bottom
                        push();
                    }).catch(error => {
                        console.error('Stream reading error:', error);
                        controller.error(error);
                    });
                }
                push();
            }
        });
    })
    .then(stream => new Response(stream).text())
    .then(result => {
        console.log('Complete response received');
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

