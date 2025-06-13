// Simulated XSS vulnerability (DO NOT USE IN PRODUCTION)

const urlParams = new URLSearchParams(window.location.search);
const userInput = urlParams.get("name");

// This line is vulnerable to reflected XSS
document.getElementById("output").innerHTML = "Hello " + userInput;
