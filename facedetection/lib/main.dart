import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:google_ml_kit/google_ml_kit.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ML Kit Face Detection',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  File? _image;
  List<Face> _detectedFaces = [];
  File? _separatedFaceImage;

  Future<void> _pickImage(ImageSource source) async {
    final imagePicker = ImagePicker();
    final pickedImage = await imagePicker.getImage(source: source);
    if (pickedImage != null) {
      setState(() {
        _image = File(pickedImage.path);
      });
      _detectFaces();
    }
  }

  Future<void> _detectFaces() async {
    final options = FaceDetectorOptions();
    final faceDetector = FaceDetector(options: options);
    final inputImage = InputImage.fromFile(_image!);
    final faces = await faceDetector.processImage(inputImage);
    setState(() {
      _detectedFaces = faces;
    });
  }

  Future<void> _separateAndStoreFaceImage(Face face) async {
    final imageBytes = await _cropFaceFromImage(face);
    final appDir = await getApplicationDocumentsDirectory();
    final filePath = '${appDir.path}/separated_face.png';
    final File separatedFaceFile = File(filePath);
    await separatedFaceFile.writeAsBytes(imageBytes);
    setState(() {
      _separatedFaceImage = separatedFaceFile;
    });
  }

  Future<Uint8List> _cropFaceFromImage(Face face) async {
    final imageBytes = await _image!.readAsBytes();
    final originalImage = img.decodeImage(imageBytes);

    final left = face.boundingBox.left.toInt();
    final top = face.boundingBox.top.toInt();
    final width = face.boundingBox.width.toInt();
    final height = face.boundingBox.height.toInt();

    final faceImage = img.copyCrop(originalImage!, x: left, y: top, width: width, height: height);
    final encodedImage = img.encodePng(faceImage);
    return encodedImage;
  }

  void _showFacePopup(Face face) async {
    final imageBytes = await _cropFaceFromImage(face);
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text('Face Detected'),
          content: Container(
            child: Image.memory(imageBytes),
          ),
          actions: [
            ElevatedButton(
              child: Text('Close'),
              onPressed: () {
                Navigator.of(context).pop();
              },
            ),
            ElevatedButton(
              child: Text('Store in Local Storage'),
              onPressed: () {
                _storeSeparatedFaceImage();
              },
            ),
            ElevatedButton(
              child: Text('Match Photo'),
              onPressed: () {
                _matchFaceWithSeparatedImage();
              },
            ),
          ],
        );
      },
    );
  }

  void _storeSeparatedFaceImage() async {
    if (_detectedFaces.isNotEmpty) {
      final selectedFace = _detectedFaces.first;
      await _separateAndStoreFaceImage(selectedFace);
      showDialog(
        context: context,
        builder: (BuildContext context) {
          return AlertDialog(
            title: Text('Image Stored'),
            content: Text('Separated face image stored successfully.'),
            actions: [
              ElevatedButton(
                child: Text('Close'),
                onPressed: () {
                  Navigator.of(context).pop();
                },
              ),
            ],
          );
        },
      );
    } else {
      showDialog(
        context: context,
        builder: (BuildContext context) {
          return AlertDialog(
            title: Text('Error'),
            content: Text('No detected faces found.'),
            actions: [
              ElevatedButton(
                child: Text('Close'),
                onPressed: () {
                  Navigator.of(context).pop();
                },
              ),
            ],
          );
        },
      );
    }
  }

  Future<double> _compareFaces(File faceImage1, File faceImage2) async {
    final faceEmbedding1 = await _getFaceEmbedding(faceImage1);
    final faceEmbedding2 = await _getFaceEmbedding(faceImage2);

    if (faceEmbedding1 == null || faceEmbedding2 == null) {
      return 0.0; 
    }

    double distance = 0;
    for (int i = 0; i < faceEmbedding1.length; i++) {
      distance += pow((faceEmbedding1[i] - faceEmbedding2[i]), 2);
    }
    distance = sqrt(distance);

    double similarity = 1 - (distance / 4);
    return similarity;
  }

  Future<List<double>> _getFaceEmbedding(File faceImage) async {
   
    return List<double>.generate(128, (index) => Random().nextDouble());
  }

  void _matchFaceWithSeparatedImage() async {
    if (_separatedFaceImage == null) {
      showDialog(
        context: context,
        builder: (BuildContext context) {
          return AlertDialog(
            title: Text('Error'),
            content: Text('Separated face image not found. Please capture and separate a face first.'),
            actions: [
              ElevatedButton(
                child: Text('Close'),
                onPressed: () {
                  Navigator.of(context).pop();
                },
              ),
            ],
          );
        },
      );
      return;
    }

    final livePhoto = await ImagePicker().getImage(source: ImageSource.camera);
    if (livePhoto != null) {
      final liveImage = File(livePhoto.path);
      final similarity = await _compareFaces(_separatedFaceImage!, liveImage);

      showDialog(
        context: context,
        builder: (BuildContext context) {
          return AlertDialog(
            title: Text('Face Match Result'),
            content: Text('Similarity Score: ${similarity.toStringAsFixed(2)}'),
            actions: [
              ElevatedButton(
                child: Text('Close'),
                onPressed: () {
                  Navigator.of(context).pop();
                },
              ),
            ],
          );
        },
      );
    }
  }

  Future<void> _removeSeparatedFaceImage() async {
    if (_separatedFaceImage != null) {
      await _separatedFaceImage!.delete();
      setState(() {
        _separatedFaceImage = null;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('ML Kit Face Detection'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            _image != null
                ? Image.file(
                    _image!,
                    height: 300,
                  )
                : Container(),
            SizedBox(height: 20),
            ElevatedButton(
              child: Text('Pick Image from Gallery'),
              onPressed: () {
                _pickImage(ImageSource.gallery);
              },
            ),
            SizedBox(height: 20),
            ElevatedButton(
              child: Text('Take Photo from Camera'),
              onPressed: () {
                _pickImage(ImageSource.camera);
              },
            ),
            SizedBox(height: 20),
            Text('Detected Faces: ${_detectedFaces.length}'),
            SizedBox(height: 10),
            Column(
              children: _detectedFaces.map<Widget>((face) {
                return ElevatedButton(
                  child: Text('Show Face'),
                  onPressed: () {
                    _showFacePopup(face);
                  },
                );
              }).toList(),
            ),
            SizedBox(height: 20),
            ElevatedButton(
              child: Text('Store in Local Storage'),
              onPressed: _storeSeparatedFaceImage,
            ),
            SizedBox(height: 20),
            ElevatedButton(
              child: Text('Remove Separated Image'),
              onPressed: _removeSeparatedFaceImage,
            ),
          ],
        ),
      ),
    );
  }
}
