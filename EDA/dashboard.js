document.addEventListener("DOMContentLoaded", () => {
    // Hardcoded metrics from the Python EDA script execution
    // (Used to bypass local file:// CORS restrictions when opening index.html directly)
    const data = {
        total_videos: 10000,
        total_captions: 17010,
        avg_fps: 27.5,
        avg_duration: 14.8
    };

    // Populate Top KPIs
    document.getElementById('metric-videos').innerText = data.total_videos.toLocaleString();
    document.getElementById('metric-captions').innerText = data.total_captions.toLocaleString();
    document.getElementById('metric-duration').innerText = data.avg_duration.toFixed(1) + "s";
    document.getElementById('metric-fps').innerText = data.avg_fps.toFixed(1);
});
