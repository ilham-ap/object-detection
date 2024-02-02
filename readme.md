{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/ilham-ap/object-detection/blob/main/object_detection_colab_webcam.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "jzRKNe1iSlNW"
      },
      "source": [
        "# Google Colab: Access Webcam for Realtime Object Detection\n",
        "This notebook will go through how to access and run code object detection using your webcam.  \n",
        "\n",
        "For this purpose of this tutorial we will be using ultralytics YOLO to do object detection on our Webcam."
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "%%capture\n",
        "!pip install ultralytics"
      ],
      "metadata": {
        "id": "d_qJt_4NKvya"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Fj9YcAnsT4B_"
      },
      "source": [
        "# import dependencies\n",
        "from IPython.display import display, Javascript, Image\n",
        "from google.colab.output import eval_js\n",
        "from base64 import b64decode, b64encode\n",
        "import cv2\n",
        "import numpy as np\n",
        "import PIL\n",
        "import io\n",
        "import html\n",
        "import time"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "L6pCmkJrUC9g"
      },
      "source": [
        "## Helper Functions\n",
        "Below are a few helper function to make converting between different image data types and formats."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "09b_0FAnUa9y"
      },
      "source": [
        "# function to convert the JavaScript object into an OpenCV image\n",
        "def js_to_image(js_reply):\n",
        "  \"\"\"\n",
        "  Params:\n",
        "          js_reply: JavaScript object containing image from webcam\n",
        "  Returns:\n",
        "          img: OpenCV BGR image\n",
        "  \"\"\"\n",
        "  # decode base64 image\n",
        "  image_bytes = b64decode(js_reply.split(',')[1])\n",
        "  # convert bytes to numpy array\n",
        "  jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)\n",
        "  # decode numpy array into OpenCV BGR image\n",
        "  img = cv2.imdecode(jpg_as_np, flags=1)\n",
        "\n",
        "  return img\n",
        "\n",
        "# function to convert OpenCV Rectangle bounding box image into base64 byte string to be overlayed on video stream\n",
        "def bbox_to_bytes(bbox_array):\n",
        "  \"\"\"\n",
        "  Params:\n",
        "          bbox_array: Numpy array (pixels) containing rectangle to overlay on video stream.\n",
        "  Returns:\n",
        "        bytes: Base64 image byte string\n",
        "  \"\"\"\n",
        "  # convert array into PIL image\n",
        "  bbox_PIL = PIL.Image.fromarray(bbox_array, 'RGBA')\n",
        "  iobuf = io.BytesIO()\n",
        "  # format bbox into png for return\n",
        "  bbox_PIL.save(iobuf, format='png')\n",
        "  # format return string\n",
        "  bbox_bytes = 'data:image/png;base64,{}'.format((str(b64encode(iobuf.getvalue()), 'utf-8')))\n",
        "\n",
        "  return bbox_bytes"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "MaZIOR4WaT64"
      },
      "source": [
        "## Haar Cascade Classifier\n",
        "For this tutorial we will run a simple object detection algorithm called Haar Cascade on our images and video fetched from our webcam. OpenCV has a pre-trained Haar Cascade face detection model."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ZpA68lTrcvZs"
      },
      "source": [
        "# initialize the Haar Cascade face detection model\n",
        "face_cascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ghUlAJzKSjFT"
      },
      "source": [
        "# JavaScript to properly create our live video stream using our webcam as input\n",
        "def video_stream():\n",
        "  js = Javascript('''\n",
        "    var video;\n",
        "    var div = null;\n",
        "    var stream;\n",
        "    var captureCanvas;\n",
        "    var imgElement;\n",
        "    var labelElement;\n",
        "\n",
        "    var pendingResolve = null;\n",
        "    var shutdown = false;\n",
        "\n",
        "    function removeDom() {\n",
        "       stream.getVideoTracks()[0].stop();\n",
        "       video.remove();\n",
        "       div.remove();\n",
        "       video = null;\n",
        "       div = null;\n",
        "       stream = null;\n",
        "       imgElement = null;\n",
        "       captureCanvas = null;\n",
        "       labelElement = null;\n",
        "    }\n",
        "\n",
        "    function onAnimationFrame() {\n",
        "      if (!shutdown) {\n",
        "        window.requestAnimationFrame(onAnimationFrame);\n",
        "      }\n",
        "      if (pendingResolve) {\n",
        "        var result = \"\";\n",
        "        if (!shutdown) {\n",
        "          captureCanvas.getContext('2d').drawImage(video, 0, 0, 640, 480);\n",
        "          result = captureCanvas.toDataURL('image/jpeg', 0.8)\n",
        "        }\n",
        "        var lp = pendingResolve;\n",
        "        pendingResolve = null;\n",
        "        lp(result);\n",
        "      }\n",
        "    }\n",
        "\n",
        "    async function createDom() {\n",
        "      if (div !== null) {\n",
        "        return stream;\n",
        "      }\n",
        "\n",
        "      div = document.createElement('div');\n",
        "      div.style.border = '2px solid black';\n",
        "      div.style.padding = '3px';\n",
        "      div.style.width = '100%';\n",
        "      div.style.maxWidth = '600px';\n",
        "      document.body.appendChild(div);\n",
        "\n",
        "      const modelOut = document.createElement('div');\n",
        "      modelOut.innerHTML = \"<span>Status:</span>\";\n",
        "      labelElement = document.createElement('span');\n",
        "      labelElement.innerText = 'No data';\n",
        "      labelElement.style.fontWeight = 'bold';\n",
        "      modelOut.appendChild(labelElement);\n",
        "      div.appendChild(modelOut);\n",
        "\n",
        "      video = document.createElement('video');\n",
        "      video.style.display = 'block';\n",
        "      video.width = div.clientWidth - 6;\n",
        "      video.setAttribute('playsinline', '');\n",
        "      video.onclick = () => { shutdown = true; };\n",
        "      stream = await navigator.mediaDevices.getUserMedia(\n",
        "          {video: { facingMode: \"environment\"}});\n",
        "      div.appendChild(video);\n",
        "\n",
        "      imgElement = document.createElement('img');\n",
        "      imgElement.style.position = 'absolute';\n",
        "      imgElement.style.zIndex = 1;\n",
        "      imgElement.onclick = () => { shutdown = true; };\n",
        "      div.appendChild(imgElement);\n",
        "\n",
        "      const instruction = document.createElement('div');\n",
        "      instruction.innerHTML =\n",
        "          '<span style=\"color: red; font-weight: bold;\">' +\n",
        "          'When finished, click here or on the video to stop this demo</span>';\n",
        "      div.appendChild(instruction);\n",
        "      instruction.onclick = () => { shutdown = true; };\n",
        "\n",
        "      video.srcObject = stream;\n",
        "      await video.play();\n",
        "\n",
        "      captureCanvas = document.createElement('canvas');\n",
        "      captureCanvas.width = 640; //video.videoWidth;\n",
        "      captureCanvas.height = 480; //video.videoHeight;\n",
        "      window.requestAnimationFrame(onAnimationFrame);\n",
        "\n",
        "      return stream;\n",
        "    }\n",
        "    async function stream_frame(label, imgData) {\n",
        "      if (shutdown) {\n",
        "        removeDom();\n",
        "        shutdown = false;\n",
        "        return '';\n",
        "      }\n",
        "\n",
        "      var preCreate = Date.now();\n",
        "      stream = await createDom();\n",
        "\n",
        "      var preShow = Date.now();\n",
        "      if (label != \"\") {\n",
        "        labelElement.innerHTML = label;\n",
        "      }\n",
        "\n",
        "      if (imgData != \"\") {\n",
        "        var videoRect = video.getClientRects()[0];\n",
        "        imgElement.style.top = videoRect.top + \"px\";\n",
        "        imgElement.style.left = videoRect.left + \"px\";\n",
        "        imgElement.style.width = videoRect.width + \"px\";\n",
        "        imgElement.style.height = videoRect.height + \"px\";\n",
        "        imgElement.src = imgData;\n",
        "      }\n",
        "\n",
        "      var preCapture = Date.now();\n",
        "      var result = await new Promise(function(resolve, reject) {\n",
        "        pendingResolve = resolve;\n",
        "      });\n",
        "      shutdown = false;\n",
        "\n",
        "      return {'create': preShow - preCreate,\n",
        "              'show': preCapture - preShow,\n",
        "              'capture': Date.now() - preCapture,\n",
        "              'img': result};\n",
        "    }\n",
        "    ''')\n",
        "\n",
        "  display(js)\n",
        "\n",
        "def video_frame(label, bbox):\n",
        "  data = eval_js('stream_frame(\"{}\", \"{}\")'.format(label, bbox))\n",
        "  return data"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "kkIPWTM5L0X0"
      },
      "source": [
        "## Ultralitics object detection yolo\n",
        "For this tutorial we will run a simple object detection algorithm ultralytics yolo from our webcam."
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from ultralytics import YOLO\n",
        "import sys\n",
        "model = YOLO(\"yolo-Weights/yolov8n.pt\")\n",
        "sys.stdout = open(\"goat.txt\", \"w\")"
      ],
      "metadata": {
        "id": "nQ9rva5u5hAg"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# object classes\n",
        "classNames = [\"person\", \"bicycle\", \"car\", \"motorbike\", \"aeroplane\", \"bus\", \"train\", \"truck\", \"boat\",\n",
        "              \"traffic light\", \"fire hydrant\", \"stop sign\", \"parking meter\", \"bench\", \"bird\", \"cat\",\n",
        "              \"dog\", \"horse\", \"sheep\", \"cow\", \"elephant\", \"bear\", \"zebra\", \"giraffe\", \"backpack\", \"umbrella\",\n",
        "              \"handbag\", \"tie\", \"suitcase\", \"frisbee\", \"skis\", \"snowboard\", \"sports ball\", \"kite\", \"baseball bat\",\n",
        "              \"baseball glove\", \"skateboard\", \"surfboard\", \"tennis racket\", \"bottle\", \"wine glass\", \"cup\",\n",
        "              \"fork\", \"knife\", \"spoon\", \"bowl\", \"banana\", \"apple\", \"sandwich\", \"orange\", \"broccoli\",\n",
        "              \"carrot\", \"hot dog\", \"pizza\", \"donut\", \"cake\", \"chair\", \"sofa\", \"pottedplant\", \"bed\",\n",
        "              \"diningtable\", \"toilet\", \"tvmonitor\", \"laptop\", \"mouse\", \"remote\", \"keyboard\", \"cell phone\",\n",
        "              \"microwave\", \"oven\", \"toaster\", \"sink\", \"refrigerator\", \"book\", \"clock\", \"vase\", \"scissors\",\n",
        "              \"teddy bear\", \"hair drier\", \"toothbrush\"\n",
        "              ]"
      ],
      "metadata": {
        "id": "dmc2HuFQ9GE5"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "lIMPmJ3UMQ09"
      },
      "source": [
        "## Let's run detection"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# start streaming video from webcam\n",
        "video_stream()\n",
        "# label for video\n",
        "label_html = 'Capturing...'\n",
        "# initialze bounding box to empty\n",
        "bbox = ''\n",
        "count = 0\n",
        "while True:\n",
        "    js_reply = video_frame(label_html, bbox)\n",
        "    if not js_reply:\n",
        "        break\n",
        "\n",
        "    # convert JS response to OpenCV Image\n",
        "    img = js_to_image(js_reply[\"img\"])\n",
        "\n",
        "    # create transparent overlay for bounding box\n",
        "    bbox_array = np.zeros([480,640,4], dtype=np.uint8)\n",
        "\n",
        "    # grayscale image for face detection\n",
        "    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)\n",
        "\n",
        "    # get face region coordinates\n",
        "    faces = model(img, stream=True)\n",
        "    print(f\"{faces} sys.stdout\")\n",
        "    # get face bounding box for overlay\n",
        "    for r in faces:\n",
        "        boxes = r.boxes\n",
        "\n",
        "        for box in boxes:\n",
        "            # bounding box\n",
        "            x1, y1, x2, y2 = box.xyxy[0]\n",
        "            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)\n",
        "            bbox_array = cv2.rectangle(bbox_array, (x1, y1), (x2, y2), (255, 0, 255), 2)\n",
        "             # class name\n",
        "            cls = int(box.cls[0])\n",
        "\n",
        "            # object details\n",
        "            org = [x1, y1]\n",
        "            font = cv2.FONT_HERSHEY_SIMPLEX\n",
        "            fontScale = 1\n",
        "            color = (255, 0, 0)\n",
        "            thickness = 2\n",
        "\n",
        "            bbox_array = cv2.putText(bbox_array, classNames[cls], org, font, fontScale, color, thickness)\n",
        "\n",
        "    bbox_array[:,:,3] = (bbox_array.max(axis = 2) > 0 ).astype(int) * 255\n",
        "    # convert overlay of bbox into bytes\n",
        "    bbox_bytes = bbox_to_bytes(bbox_array)\n",
        "    # update bbox so next frame gets new overlay\n",
        "    bbox = bbox_bytes"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 17
        },
        "id": "uUhSGqVl4aA4",
        "outputId": "69f133f4-1fc8-4812-bf51-2f04273f470f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.Javascript object>"
            ],
            "application/javascript": [
              "\n",
              "    var video;\n",
              "    var div = null;\n",
              "    var stream;\n",
              "    var captureCanvas;\n",
              "    var imgElement;\n",
              "    var labelElement;\n",
              "\n",
              "    var pendingResolve = null;\n",
              "    var shutdown = false;\n",
              "\n",
              "    function removeDom() {\n",
              "       stream.getVideoTracks()[0].stop();\n",
              "       video.remove();\n",
              "       div.remove();\n",
              "       video = null;\n",
              "       div = null;\n",
              "       stream = null;\n",
              "       imgElement = null;\n",
              "       captureCanvas = null;\n",
              "       labelElement = null;\n",
              "    }\n",
              "\n",
              "    function onAnimationFrame() {\n",
              "      if (!shutdown) {\n",
              "        window.requestAnimationFrame(onAnimationFrame);\n",
              "      }\n",
              "      if (pendingResolve) {\n",
              "        var result = \"\";\n",
              "        if (!shutdown) {\n",
              "          captureCanvas.getContext('2d').drawImage(video, 0, 0, 640, 480);\n",
              "          result = captureCanvas.toDataURL('image/jpeg', 0.8)\n",
              "        }\n",
              "        var lp = pendingResolve;\n",
              "        pendingResolve = null;\n",
              "        lp(result);\n",
              "      }\n",
              "    }\n",
              "\n",
              "    async function createDom() {\n",
              "      if (div !== null) {\n",
              "        return stream;\n",
              "      }\n",
              "\n",
              "      div = document.createElement('div');\n",
              "      div.style.border = '2px solid black';\n",
              "      div.style.padding = '3px';\n",
              "      div.style.width = '100%';\n",
              "      div.style.maxWidth = '600px';\n",
              "      document.body.appendChild(div);\n",
              "\n",
              "      const modelOut = document.createElement('div');\n",
              "      modelOut.innerHTML = \"<span>Status:</span>\";\n",
              "      labelElement = document.createElement('span');\n",
              "      labelElement.innerText = 'No data';\n",
              "      labelElement.style.fontWeight = 'bold';\n",
              "      modelOut.appendChild(labelElement);\n",
              "      div.appendChild(modelOut);\n",
              "\n",
              "      video = document.createElement('video');\n",
              "      video.style.display = 'block';\n",
              "      video.width = div.clientWidth - 6;\n",
              "      video.setAttribute('playsinline', '');\n",
              "      video.onclick = () => { shutdown = true; };\n",
              "      stream = await navigator.mediaDevices.getUserMedia(\n",
              "          {video: { facingMode: \"environment\"}});\n",
              "      div.appendChild(video);\n",
              "\n",
              "      imgElement = document.createElement('img');\n",
              "      imgElement.style.position = 'absolute';\n",
              "      imgElement.style.zIndex = 1;\n",
              "      imgElement.onclick = () => { shutdown = true; };\n",
              "      div.appendChild(imgElement);\n",
              "\n",
              "      const instruction = document.createElement('div');\n",
              "      instruction.innerHTML =\n",
              "          '<span style=\"color: red; font-weight: bold;\">' +\n",
              "          'When finished, click here or on the video to stop this demo</span>';\n",
              "      div.appendChild(instruction);\n",
              "      instruction.onclick = () => { shutdown = true; };\n",
              "\n",
              "      video.srcObject = stream;\n",
              "      await video.play();\n",
              "\n",
              "      captureCanvas = document.createElement('canvas');\n",
              "      captureCanvas.width = 640; //video.videoWidth;\n",
              "      captureCanvas.height = 480; //video.videoHeight;\n",
              "      window.requestAnimationFrame(onAnimationFrame);\n",
              "\n",
              "      return stream;\n",
              "    }\n",
              "    async function stream_frame(label, imgData) {\n",
              "      if (shutdown) {\n",
              "        removeDom();\n",
              "        shutdown = false;\n",
              "        return '';\n",
              "      }\n",
              "\n",
              "      var preCreate = Date.now();\n",
              "      stream = await createDom();\n",
              "\n",
              "      var preShow = Date.now();\n",
              "      if (label != \"\") {\n",
              "        labelElement.innerHTML = label;\n",
              "      }\n",
              "\n",
              "      if (imgData != \"\") {\n",
              "        var videoRect = video.getClientRects()[0];\n",
              "        imgElement.style.top = videoRect.top + \"px\";\n",
              "        imgElement.style.left = videoRect.left + \"px\";\n",
              "        imgElement.style.width = videoRect.width + \"px\";\n",
              "        imgElement.style.height = videoRect.height + \"px\";\n",
              "        imgElement.src = imgData;\n",
              "      }\n",
              "\n",
              "      var preCapture = Date.now();\n",
              "      var result = await new Promise(function(resolve, reject) {\n",
              "        pendingResolve = resolve;\n",
              "      });\n",
              "      shutdown = false;\n",
              "\n",
              "      return {'create': preShow - preCreate,\n",
              "              'show': preCapture - preShow,\n",
              "              'capture': Date.now() - preCapture,\n",
              "              'img': result};\n",
              "    }\n",
              "    "
            ]
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "vCBVkX5mNukp"
      },
      "source": [
        "## Hope You Enjoyed!\n",
        "If you enjoyed the tutorial and want to see more videos or tutorials check out my github [HERE](https://github.com/ilham-ap)\n",
        "\n",
        "Have a great day!"
      ]
    }
  ]
}
