const usernameSpan = document.getElementById('username');


function changeUsername() {
    // Prompt user for new username
    let newUsername = prompt("Enter new username:");

    fetch('/api/auth/change-username', {
        method: 'POST',
        body: JSON.stringify({new_username: newUsername}),
        headers: {'Content-Type': 'application/json'}
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            usernameSpan.innerText = newUsername;
            alert('Username changed successfully!');
        }
        else {alert(data.message);}
    });
}