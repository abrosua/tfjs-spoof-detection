# Face anti-spoofing detection using TensorflowJS

## Contents

This demo shows how to use the Blazeface model to detect faces in a video stream. Then the face image(s) will be cropped and parsed to the classification model

## Setup

First, you need to install [Node.js](https://nodejs.org/en/) and [yarn](https://classic.yarnpkg.com/en/docs/install/#debian-stable) on your machine, with the corresponding operating system, to build and/or run the app.
 
Install dependencies and prepare the build directory:

```sh
yarn
```

Launching the development server (by default, it's running at http://localhost:1234) :

```sh
yarn watch
```

Build the app:

```sh
yarn build
```
This will create a `dist/` folder, which contains the static files needed to serve the app.

After finish building the app, you could deploy the app using [Firebase hosting](https://console.firebase.google.com/), using the following command.
```shell script
firebase deploy
```

> You need to login into your firebase account first and create your own hosting, refer to the firebase website for the complete guide.

## Demo
We already deploy this app using a firebase hosting, and you could access it using this [link](bit.ly/spoof-detection).