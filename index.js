/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as blazeface from '@tensorflow-models/blazeface';
import * as tf from '@tensorflow/tfjs-core';
import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';

tfjsWasm.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@latest/dist/tfjs-backend-wasm.wasm');

const stats = new Stats();
stats.showPanel(0);
document.body.prepend(stats.domElement);

let model, classifier, ctx, videoWidth, videoHeight, video, videoCrop, canvas;

const state = {
  backend: 'wasm'
};

const gui = new dat.GUI();
gui.add(state, 'backend', ['wasm', 'webgl', 'cpu']).onChange(async backend => {
  await tf.setBackend(backend);
});

async function setupCamera() {
  video = document.getElementById('video');

  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': { facingMode: 'user' },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

const renderPrediction = async () => {
  stats.begin();
  const font = "16px sans-serif";
  ctx.font = font;

  const returnTensors = false;
  const flipHorizontal = true;
  const annotateBoxes = true;
  const classifySpoof = false;

  const predictions = await model.estimateFaces(
    video, returnTensors, flipHorizontal, annotateBoxes);

  if (predictions.length > 0) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let i = 0; i < predictions.length; i++) {
      if (returnTensors) {
        predictions[i].topLeft = predictions[i].topLeft.arraySync();
        predictions[i].bottomRight = predictions[i].bottomRight.arraySync();
      }

      const start = predictions[i].topLeft;
      const end = predictions[i].bottomRight;
      const size = [end[0] - start[0], end[1] - start[1]];
      const mid = [(start[0] + end[0]) * 0.5, (start[1] + end[1]) * 0.5]

      // create a Square bounding box
      const scale = 1
      const sizeNew = Math.max(size[0], size[1]) * scale
      const startNew = [mid[0] - (sizeNew * 0.5), mid[1] - (sizeNew * 0.5)]

      // Rendering the bounding box
      ctx.strokeStyle="red";
      ctx.lineWidth = "4";
      ctx.strokeRect(startNew[0], startNew[1], sizeNew, sizeNew);

      // Perform spoof classification (UNFINISHED!)
      if (classifySpoof) {
        // Cropping the frame and perform spoof classification
        const endNew = [startNew[0] + sizeNew, startNew[1] + sizeNew]
        videoCrop = video.slice([startNew[0], startNew[1], 0], [endNew[0], endNew[1], 3])
        const labelPredict = classifier(videoCrop)

        if (labelPredict < 0.5) {
          label = "Real"
        } else if (labelPredict > 0.5) {
          label = "Spoof"
        } else {
          label = "ERROR!"
        }

        // Drawing the label
        ctx.fillStyle = "red";
        const textWidth = ctx.measureText(label).width;
        const textHeight = parseInt(font, 10); // base 10
        ctx.fillRect(startNew[0], startNew[1], textWidth + 4, textHeight + 4);

        ctx.fillStyle = "#000000";
        ctx.fillText(label, startNew[0], startNew[1]);
      }
    }
  }

  stats.end();

  requestAnimationFrame(renderPrediction);
};

const setupPage = async () => {
  await tf.setBackend(state.backend);
  await setupCamera();
  video.play();

  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;
  video.width = videoWidth;
  video.height = videoHeight;

  canvas = document.getElementById('output');
  canvas.width = videoWidth;
  canvas.height = videoHeight;
  ctx = canvas.getContext('2d');
  ctx.fillStyle = "rgba(255, 0, 0, 0.5)";

  model = await blazeface.load();
  // classifier = await tf.loadModel('./models/mobilenet-spoof/model.json')

  renderPrediction();
};

setupPage();
