const cam = document.getElementById('cam')

// acessar câmera específica:
const startVideo = () => {
    navigator.mediaDevices.enumerateDevices()
    .then(devices => {
        if (Array.isArray(devices)) {
            // tem dispositivos...
            devices.forEach(device => {
                if (device.kind === 'videoinput') {
                    // é uma câmera...
                    if (device.label.includes('')) {
                        // se a label conter o texto mencionado no apóstrofo, entra no if
                        // feito para selecionar qual câmera irá ser utilizada
                        // acesso à câmera:
                        navigator.getUserMedia(
                            { video: {
                                deviceId: device.deviceId
                            }},
                            stream => cam.srcObject = stream,
                            error => console.log(error)
                        )
                    }
                }
            })
        }
    })
}

const loadLabels = () => {
    const labels = ['Leonardo Aoki', 'Luan Natan', 'Luiz Ciantela']
    return Promise.all(labels.map(async label => {
        const descriptions = []
        for (let i = 1; i <= 5; i++) {
            const img = await faceapi.fetchImage(`/assets/lib/face-api/labels/${label}/${i}.jpg`)
            const detections = await faceapi
                .detectSingleFace(img)
                .withFaceLandmarks()
                .withFaceDescriptor()
            descriptions.push(detections.descriptor)
        }
        return new faceapi.LabeledFaceDescriptors(label, descriptions)
    }))
}

// importando redes neurais da face-api:
Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri('/assets/lib/face-api/models'), // detectar rostos no vídeo
    faceapi.nets.faceLandmark68Net.loadFromUri('/assets/lib/face-api/models'), // desenhar os traços no rosto (boca, nariz, olhos)
    faceapi.nets.faceRecognitionNet.loadFromUri('/assets/lib/face-api/models'), // reconhecer o rosto (quem é)
    faceapi.nets.faceExpressionNet.loadFromUri('/assets/lib/face-api/models'), // detectar expressões
    faceapi.nets.ageGenderNet.loadFromUri('/assets/lib/face-api/models'), // detectar idade e gênero
    faceapi.nets.ssdMobilenetv1.loadFromUri('/assets/lib/face-api/models'), // detectar rostos (internamente)
]).then(startVideo)

cam.addEventListener('play', async () => {
    const canvas = faceapi.createCanvasFromMedia(cam)
    const canvasSize = {
        width: cam.width,
        height: cam.height
    }
    const labels = await loadLabels()
    faceapi.matchDimensions(canvas, canvasSize)
    document.body.appendChild(canvas)
    setInterval(async () => {
        const detections = await faceapi
            .detectAllFaces(
                cam,
                new faceapi.TinyFaceDetectorOptions()
            )
            .withFaceLandmarks()
            .withFaceExpressions()
            .withAgeAndGender()
            .withFaceDescriptors()
        const resizedDetections = faceapi.resizeResults(detections, canvasSize)
        const faceMatcher = new faceapi.FaceMatcher(labels, 0.6)
        const results = resizedDetections.map(d => 
            faceMatcher.findBestMatch(d.descriptor)    
        )
        canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)
        faceapi.draw.drawDetections(canvas, resizedDetections)
        faceapi.draw.drawFaceLandmarks(canvas, resizedDetections)
        faceapi.draw.drawFaceExpressions(canvas, resizedDetections)
        resizedDetections.forEach(detection => {
            const { age, gender, genderProbability } = detection
            new faceapi.draw.DrawTextField([
                `${parseInt(age, 10)} years`,
                `${gender} (${parseInt(genderProbability * 100, 10)})`
            ], detection.detection.box.topRight).draw(canvas)
        })
        results.forEach((result, index) => {
            const box = resizedDetections[index].detection.box
            const { label, distance } = result
            new faceapi.draw.DrawTextField([
                `${label} (${parseInt(distance * 100, 10)})`
            ], box.bottomRight).draw(canvas)
        })
    }, 100)
})