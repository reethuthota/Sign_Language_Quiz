// Start button click handler
document.getElementById('start-button').onclick = function() {
    startNewWord();
};

// Next word button click handler
document.getElementById('next-word-button').onclick = function() {
    startNewWord();
};

function startNewWord() {
    // Reset UI
    document.getElementById('start-button').style.display = 'none';
    document.getElementById('target-section').style.display = 'block';
    document.getElementById('result').style.display = 'none';
    document.getElementById('timer').style.display = 'block';
    document.getElementById('timer').textContent = 'Get ready!';
    document.getElementById('next-word-button').style.display = 'none';

    // Reset quiz state on server
    fetch('/reset_quiz_state')
        .then(() => {
            // Start the camera feed
            startVideoFeed();

            // Fetch new target word
            return fetch('/get_target_word');
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('word').textContent = data.target;

            // Start timer after 5-second delay
            setTimeout(() => {
                startTimer(30);
            }, 5000);
        });
}

function startVideoFeed() {
    const videoElement = document.getElementById('video');
    videoElement.src = '/video_feed';
    document.getElementById('video-container').style.display = 'block';
}

function startTimer(duration) {
    let timer = duration;
    const interval = setInterval(function() {
        document.getElementById('timer').textContent = timer;
        
        fetch('/check_for_correct')
            .then(response => response.json())
            .then(data => {
                if (data.quiz_ended) {
                    clearInterval(interval);
                    stopVideoFeed();
                    if (data.timer_expired) {
                        document.getElementById('result-text').textContent = 'Time Up! Try another word!';
                    } else {
                        document.getElementById('result-text').textContent = 'Correct!';
                    }
                    document.getElementById('result').style.display = 'block';
                    document.getElementById('next-word-button').style.display = 'block';
                }
            });

        if (timer <= 0) {
            clearInterval(interval);
            fetch('/stop_quiz')
                .then(response => response.json())
                .then(() => {
                    stopVideoFeed();
                    document.getElementById('result-text').textContent = 'Time Up! Try another word!';
                    document.getElementById('result').style.display = 'block';
                    document.getElementById('next-word-button').style.display = 'block';
                });
        } else {
            timer--;
        }
    }, 1000);
}

function stopVideoFeed() {
    const videoElement = document.getElementById('video');
    videoElement.src = '';
    document.getElementById('video-container').style.display = 'none';
}