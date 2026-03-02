async function analyzeVideo() {
    const fileInput = document.getElementById("videoInput");
    const resultDiv = document.getElementById("result");

    if (!fileInput.files.length) {
        alert("Please select a video.");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    resultDiv.innerHTML = "Analyzing... Please wait.";

    try {
        const response = await fetch("http://127.0.0.1:8080/detect/", {
            method: "POST",
            body: formData
        });

        const data = await response.json();
        console.log("Backend Response:", data);

        if (data.error) {
            resultDiv.innerHTML = "Error: " + data.error;
            return;
        }

        if (data.result && data.confidence !== undefined) {
            resultDiv.innerHTML = `
                <p>Result: <strong>${data.result}</strong></p>
                <p>Confidence: <strong>${data.confidence}%</strong></p>
            `;
        } else {
            resultDiv.innerHTML = "Unexpected response from server.";
        }

    } catch (error) {
        console.error(error);
        resultDiv.innerHTML = "Server connection failed.";
    }
}