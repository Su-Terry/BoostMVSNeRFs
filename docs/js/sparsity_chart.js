let lastHoveredxLabel = null;
const xValues = [10, 20, 25, 50, 75, 100];

new Chart("sparsityChart", {
    type: "line",
    data: {
        labels: xValues,
        datasets: [
            { 
                label: 'ENeRF+ours',
                data: [20.21, 22.04, 23.44, 25.56, 26.39, 27.54],
                borderColor: "#66b266",
                backgroundColor: '#329932',
                fill: false,
                tension: 0,
                pointRadius: 5,
            },
            { 
                label: 'ENeRF',
                data: [18.88, 20.75, 21.7, 24.99, 26.05, 27.05],
                borderColor: "#a6a6a6",
                backgroundColor: '#8c8c8c',
                fill: false,
                tension: 0,
                pointRadius: 5,
            }
        ]
    },
    options: {
        legend: {display: true},
        maintainAspectRatio: false,
        scales: {
            yAxes: [{
              scaleLabel: {
                display: true,
                labelString: 'PSNR'
              }
            }],
            xAxes: [{
              scaleLabel: {
                display: true,
                labelString: '% of input views'
              }
            }]
        },
        tooltips: {
            mode: 'index',
            intersect: false,
            callbacks: {
                label: function(tooltipItem, data) {
                    // Get the hovered data point values
                    const datasetLabel = data.datasets[tooltipItem.datasetIndex].label;
                    const value = data.datasets[tooltipItem.datasetIndex].data[tooltipItem.index];
                    const xLabel = data.labels[tooltipItem.index];
                    
                    // Call your custom function with the values
                    // customHoverFunction(xLabel);
                    // if (lastHoveredxLabel != xLabel){
                    //     var video = document.getElementById("sparsityVideo");
                    //     video.src = "videos/sparsity/kitchenlego_" + xLabel + ".mp4";
                    //     lastHoveredxLabel = xLabel;
                    //     var video_label = document.getElementById("sparsityValue");
                    //     video_label.innerHTML = xLabel;
                    // }
                    
                    // Return the tooltip label text
                    return datasetLabel + ': ' + value;
                }
            }
        }
    }
});


function playVideo(videoId, targetHeight=400) {
    var videoMerge = document.getElementById(videoId + "Wrapper");
    var vid = document.getElementById(videoId);

    var mergeContext = videoMerge.getContext("2d");
    var vidWidthOrig = vid.videoWidth;
    var vidHeightOrig = vid.videoHeight;
    var vidWidth = targetHeight / vid.videoHeight * vid.videoWidth;
    var vidHeight = targetHeight;

    
    if (vid.readyState > 3) {
        vid.play();
        function drawLoop() {
            mergeContext.drawImage(vid, 0, 0, vidWidthOrig, vidHeightOrig, 0, 0, vidWidth, vidHeight);
            requestAnimationFrame(drawLoop);
        }
        requestAnimationFrame(drawLoop);
    } 
}

// wrapper for video to avoid flickering after changing source
function playOnCanvas(element, targetHeight=400)
{
    var cv = document.getElementById(element.id + "Wrapper");
    cv.width = element.videoWidth;
    cv.height = element.videoHeight;
    
    cv.width = targetHeight / element.videoHeight * element.videoWidth;
    cv.height = targetHeight;
    element.play();
    element.style.height = "0px";  // Hide video without stopping it
        
    playVideo(element.id, targetHeight);
}
