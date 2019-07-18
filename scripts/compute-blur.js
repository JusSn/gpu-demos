async function computeBlur() {
    if (!navigator.gpu) {
        document.body.className = 'error';
        return;
    }

    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();

    const canvas = document.querySelector("canvas");
    const context2d = canvas.getContext("2d");

    /* Image */
    const image = new Image();
    const imageLoadPromise = new Promise(resolve => { 
        image.onload = () => resolve(); 
        image.src = "resources/blue-checkered.png"
    });
    await Promise.resolve(imageLoadPromise);

    canvas.height = image.height;
    canvas.width = image.width;

    context2d.drawImage(image, 0, 0);
}

window.addEventListener("load", computeBlur);