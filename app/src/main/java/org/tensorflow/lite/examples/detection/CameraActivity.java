/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.Image;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.Surface;
import android.view.View;
import android.view.ViewTreeObserver;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.content.ContextCompat;
import androidx.databinding.DataBindingUtil;

import com.google.android.material.bottomsheet.BottomSheetBehavior;
import com.google.common.util.concurrent.ListenableFuture;

import org.tensorflow.lite.examples.detection.databinding.TfeOdActivityCameraBinding;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Detector;
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutionException;

public class CameraActivity extends AppCompatActivity implements View.OnClickListener {
  private static final Logger LOGGER = new Logger();

  private static final int PERMISSIONS_REQUEST = 1;

  private static final String PERMISSION_CAMERA = Manifest.permission.CAMERA;
  protected int previewWidth = 0;
  protected int previewHeight = 0;

  private boolean debug = false;

  private Handler handler;
  private HandlerThread handlerThread;
  private boolean firstTimeStartModel = true;
  private boolean isProcessingFrame = false;

  private LinearLayout bottomSheetLayout;
  private BottomSheetBehavior<LinearLayout> sheetBehavior;

  private enum DetectorMode {
    TF_OD_API;
  }

  private static final int TF_OD_API_INPUT_SIZE = 300;
  private static final boolean TF_OD_API_IS_QUANTIZED = true;
  private static final String TF_OD_API_MODEL_FILE = "detect.tflite";
  private static final String TF_OD_API_LABELS_FILE = "labelmap.txt";
  private static final DetectorMode MODE = DetectorMode.TF_OD_API;

  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
  private static final boolean MAINTAIN_ASPECT = false;
  private static final Size DESIRED_ANALYSIS_SIZE = new Size(640, 480);
  private static final float TEXT_SIZE_DIP = 10;
  private long lastProcessingTimeMs;

  protected ImageView bottomSheetArrowImageView;

  private Integer sensorOrientation;

  private Detector detector;

  //Data Binding
  private TfeOdActivityCameraBinding binding;

  //private Bitmap cropCopyBitmap = null;
  private byte[][] yuvBytes = new byte[3][];
  private int[] rgbBytes = null;
  private int yRowStride;

  static final int kMaxChannelValue = 262143;
  private Matrix frameToCropTransform;
  //private Matrix cropToFrameTransform;
  private long timestamp = 0;
  private MultiBoxTracker tracker;


  @Override
  protected void onCreate(final Bundle savedInstanceState) {
    LOGGER.d("onCreate " + this);
    super.onCreate(null);
    getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

    binding = DataBindingUtil.setContentView(this, R.layout.tfe_od_activity_camera);

    if (hasPermission()) {
      //Start CameraX
      startCamera();
    } else {
      requestPermission();
    }

    bottomSheetLayout = findViewById(R.id.bottom_sheet_layout);
    sheetBehavior = BottomSheetBehavior.from(bottomSheetLayout);
    bottomSheetArrowImageView = findViewById(R.id.bottom_sheet_arrow);

    //Controlling bottom modal sheet
    ViewTreeObserver vto = binding.bottomSheetLayout.gestureLayout.getViewTreeObserver();
    vto.addOnGlobalLayoutListener(
            new ViewTreeObserver.OnGlobalLayoutListener() {
              @Override
              public void onGlobalLayout() {
                binding.bottomSheetLayout.gestureLayout.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                int height = binding.bottomSheetLayout.gestureLayout.getMeasuredHeight();
                sheetBehavior.setPeekHeight(height);
              }
            });
    sheetBehavior.setHideable(false);

    sheetBehavior.setBottomSheetCallback(
            new BottomSheetBehavior.BottomSheetCallback() {
              @Override
              public void onStateChanged(@NonNull View bottomSheet, int newState) {
                switch (newState) {
                  case BottomSheetBehavior.STATE_HIDDEN:
                    break;
                  case BottomSheetBehavior.STATE_EXPANDED: {
                    bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_down);
                  }
                  break;
                  case BottomSheetBehavior.STATE_COLLAPSED: {
                    bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_up);
                  }
                  break;
                  case BottomSheetBehavior.STATE_DRAGGING:
                    break;
                  case BottomSheetBehavior.STATE_SETTLING:
                    bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_up);
                    break;
                }
              }

              @Override
              public void onSlide(@NonNull View bottomSheet, float slideOffset) {
              }
            });


    binding.bottomSheetLayout.plus.setOnClickListener(this);
    binding.bottomSheetLayout.minus.setOnClickListener(this);
  }

  private void onStartCameraX(Size size, final int rotation) {
    final float textSizePx =
            TypedValue.applyDimension(
                    TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    BorderedText borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);
    int cropSize = TF_OD_API_INPUT_SIZE;
    Log.v("Camera oImageRotation", String.valueOf(rotation));
    Log.v("Camera oScreenOrientati", String.valueOf(getScreenOrientation()));
    sensorOrientation = rotation;// - getScreenOrientation();
    previewWidth = size.getWidth();
    previewHeight = size.getHeight();
    //cropToFrameTransform = new Matrix();
    /*frameToCropTransform = getTransformationMatrix(
            previewWidth, previewHeight,
            640, 480,
            0, false);*/
    //frameToCropTransform.invert(cropToFrameTransform);
    tracker = new MultiBoxTracker(this);

    try {
      detector =
              TFLiteObjectDetectionAPIModel.create(
                      this,
                      TF_OD_API_MODEL_FILE,
                      TF_OD_API_LABELS_FILE,
                      TF_OD_API_INPUT_SIZE,
                      TF_OD_API_IS_QUANTIZED);
      cropSize = TF_OD_API_INPUT_SIZE;
    } catch (final IOException e) {
      e.printStackTrace();
      LOGGER.e(e, "Exception initializing Detector!");
      Toast toast =
              Toast.makeText(
                      getApplicationContext(), "Detector could not be initialized", Toast.LENGTH_SHORT);
      toast.show();
      finish();
    }

    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);
    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);

    binding.trackingOverlay.addCallback(
            canvas -> {
              tracker.draw(canvas);
              if (isDebug()) {
                tracker.drawDebug(canvas);
              }
            });

    tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
  }

  @SuppressLint("UnsafeOptInUsageError")
  private void startCamera() {
    ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);

    cameraProviderFuture.addListener(() -> {
      try {
        ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
        Preview preview = new Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .build();

        //Selecting the Camera here - Back Camera
        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();

        //Images are processed by passing an executor in which the image analysis is run
        ImageAnalysis imageAnalysis =
                new ImageAnalysis.Builder()
                        .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

        imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(this), image -> {
          int rotationDegrees = image.getImageInfo().getRotationDegrees();
          Log.i("Rotation Degrees", String.valueOf(rotationDegrees));
          Log.i("Rotation preview", String.valueOf(binding.previewView.getDisplay().getRotation()));
          Log.i("Image width", String.valueOf(image.getWidth()));
          Log.i("Image height", String.valueOf(image.getHeight()));

          ++timestamp;
          final long currTimestamp = timestamp;

          if (firstTimeStartModel) {
            onStartCameraX(DESIRED_ANALYSIS_SIZE, rotationDegrees);
            firstTimeStartModel = false;
          }

          if (!isProcessingFrame) {
            //final int cropSize = Math.min(DESIRED_PREVIEW_SIZE.getWidth(), DESIRED_PREVIEW_SIZE.getHeight());

            runInBackground(
                    () -> {
                      if (detector != null) {
                        final long startTime = SystemClock.uptimeMillis();
                        final List<Detector.Recognition> results = detector.recognizeImage(image.getImage(), sensorOrientation);
                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                        LOGGER.e("Degrees: %s", results);

                        //cropCopyBitmap = Bitmap.createBitmap(imageToRGB(image.getImage(), image.getWidth(), image.getHeight()));
                        //File photoFile = createFile(this, "jpg");
                        //File filePath = saveBitmap(cropCopyBitmap, photoFile);
                        //Log.v("File_path", filePath.toString());
                        /*final Canvas canvas = new Canvas(cropCopyBitmap);
                        final Paint paint = new Paint();
                        paint.setColor(Color.RED);
                        paint.setStyle(Paint.Style.STROKE);
                        paint.setStrokeWidth(2.0f);*/

                        float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                        if (MODE == DetectorMode.TF_OD_API) {
                          minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                        }

                        final List<Detector.Recognition> mappedRecognitions =
                                new ArrayList<>();

                        for (final Detector.Recognition result : results) {
                          final RectF location = result.getLocation();
                          if (location != null && result.getConfidence() >= minimumConfidence) {
                            //canvas.drawRect(location, paint);

                            //frameToCropTransform.mapRect(location);

                            result.setLocation(location);
                            mappedRecognitions.add(result);
                          }
                        }

                        tracker.trackResults(mappedRecognitions, currTimestamp);
                        binding.trackingOverlay.postInvalidate();

                        runOnUiThread(
                                () -> {
                                  showFrameInfo(DESIRED_ANALYSIS_SIZE.getWidth() + "x" + DESIRED_ANALYSIS_SIZE.getHeight());
                                  showCropInfo(TF_OD_API_INPUT_SIZE + "x" + TF_OD_API_INPUT_SIZE);
                                  showInference(lastProcessingTimeMs + "ms");
                                });
                      }
                      image.close();
                      isProcessingFrame = false;
                    });
            isProcessingFrame = true;
          }
        });

        // Connect the preview use case to the previewView
        preview.setSurfaceProvider(binding.previewView.getSurfaceProvider());

        // Attach use cases to the camera with the same lifecycle owner
        if (cameraProvider != null) {
          Camera camera = cameraProvider.bindToLifecycle(
                  this,
                  cameraSelector,
                  imageAnalysis,
                  preview);
        }

      } catch (ExecutionException | InterruptedException e) {
        e.printStackTrace();
      }
    }, ContextCompat.getMainExecutor(this));
  }

  private File createFile(Context context, String extension) {
    SimpleDateFormat sdf = new SimpleDateFormat("yyyy_MM_dd_HH_mm_ss_SSS", Locale.US);
    return new File(context.getFilesDir(), "IMG_" + sdf.format(new Date()) + '.' + extension);
  }

  private File saveBitmap(@Nullable Bitmap bitmap, File file) {

    try {
      OutputStream stream = (OutputStream) (new FileOutputStream(file));
      if (bitmap != null) {
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, stream);
      }

      stream.flush();
      stream.close();
    } catch (IOException var4) {
      var4.printStackTrace();
    }

    return file;
  }

  public boolean isDebug() {
    return debug;
  }

  @Override
  public synchronized void onStart() {
    LOGGER.d("onStart " + this);
    super.onStart();
  }

  @Override
  public synchronized void onResume() {
    LOGGER.d("onResume " + this);
    super.onResume();

    handlerThread = new HandlerThread("inference");
    handlerThread.start();
    handler = new Handler(handlerThread.getLooper());
  }

  @Override
  public synchronized void onPause() {
    LOGGER.d("onPause " + this);

    handlerThread.quitSafely();
    try {
      handlerThread.join();
      handlerThread = null;
      handler = null;
    } catch (final InterruptedException e) {
      LOGGER.e(e, "Exception!");
    }

    super.onPause();
  }

  @Override
  public synchronized void onStop() {
    LOGGER.d("onStop " + this);
    super.onStop();
  }

  @Override
  public synchronized void onDestroy() {
    LOGGER.d("onDestroy " + this);
    super.onDestroy();
  }

  protected synchronized void runInBackground(final Runnable r) {
    if (handler != null) {
      handler.post(r);
    }
  }


  @Override
  public void onRequestPermissionsResult(
          final int requestCode, final String[] permissions, final int[] grantResults) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    if (requestCode == PERMISSIONS_REQUEST) {
      if (allPermissionsGranted(grantResults)) {
        //Start CameraX
        startCamera();
      } else {
        requestPermission();
      }
    }
  }

  private static boolean allPermissionsGranted(final int[] grantResults) {
    for (int result : grantResults) {
      if (result != PackageManager.PERMISSION_GRANTED) {
        return false;
      }
    }
    return true;
  }

  private boolean hasPermission() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
      return checkSelfPermission(PERMISSION_CAMERA) == PackageManager.PERMISSION_GRANTED;
    } else {
      return true;
    }
  }

  private void requestPermission() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
      if (shouldShowRequestPermissionRationale(PERMISSION_CAMERA)) {
        Toast.makeText(
                CameraActivity.this,
                "Camera permission is required for this demo",
                Toast.LENGTH_LONG)
                .show();
      }
      requestPermissions(new String[]{PERMISSION_CAMERA}, PERMISSIONS_REQUEST);
    }
  }

  protected int getScreenOrientation() {
    switch (getWindowManager().getDefaultDisplay().getRotation()) {
      case Surface.ROTATION_270:
        return 270;
      case Surface.ROTATION_180:
        return 180;
      case Surface.ROTATION_90:
        return 90;
      default:
        return 0;
    }
  }

  protected void showFrameInfo(String frameInfo) {
    binding.bottomSheetLayout.frameInfo.setText(frameInfo);
  }

  protected void showCropInfo(String cropInfo) {
    binding.bottomSheetLayout.cropInfo.setText(cropInfo);
  }

  protected void showInference(String inferenceTime) {
    binding.bottomSheetLayout.inferenceInfo.setText(inferenceTime);
  }

  private void setNumThreads(int numThreads) throws IOException {
    LOGGER.d("Updating  numThreads: " + numThreads);
    detector.setNumThreads(this, TF_OD_API_LABELS_FILE, numThreads);
  }

  @Override
  public void onClick(View v) {
    if (v.getId() == R.id.plus) {
      String threads = binding.bottomSheetLayout.threads.getText().toString().trim();
      int numThreads = Integer.parseInt(threads);
      if (numThreads >= 9) return;
      numThreads++;
      binding.bottomSheetLayout.threads.setText(String.valueOf(numThreads));
      try {
        setNumThreads(numThreads);
      } catch (IOException e) {
        e.printStackTrace();
      }
    } else if (v.getId() == R.id.minus) {
      String threads = binding.bottomSheetLayout.threads.getText().toString().trim();
      int numThreads = Integer.parseInt(threads);
      if (numThreads == 1) {
        return;
      }
      numThreads--;
      binding.bottomSheetLayout.threads.setText(String.valueOf(numThreads));
      try {
        setNumThreads(numThreads);
      } catch (IOException e) {
        e.printStackTrace();
      }
    }
  }

  private Bitmap imageToRGB(final Image image, final int width, final int height) {
    if (rgbBytes == null) {
      rgbBytes = new int[width * height];
    }

    Bitmap rgbFrameBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);

    try {

      if (image == null) {
        return null;
      }

      Log.e("Degrees_length", String.valueOf(rgbBytes.length));
      final Image.Plane[] planes = image.getPlanes();
      fillBytesCameraX(planes, yuvBytes);
      yRowStride = planes[0].getRowStride();
      final int uvRowStride = planes[1].getRowStride();
      final int uvPixelStride = planes[1].getPixelStride();

      convertYUV420ToARGB8888(
              yuvBytes[0],
              yuvBytes[1],
              yuvBytes[2],
              width,
              height,
              yRowStride,
              uvRowStride,
              uvPixelStride,
              rgbBytes);


      rgbFrameBitmap.setPixels(rgbBytes, 0, width, 0, 0, width, height);


    } catch (final Exception e) {
      Log.e(e.toString(), "Exception!");
    }

    return rgbFrameBitmap;
  }

  private void fillBytesCameraX(final Image.Plane[] planes, final byte[][] yuvBytes) {
    // Because of the variable row stride it's not possible to know in
    // advance the actual necessary dimensions of the yuv planes.
    for (int i = 0; i < planes.length; ++i) {
      final ByteBuffer buffer = planes[i].getBuffer();
      if (yuvBytes[i] == null) {
        yuvBytes[i] = new byte[buffer.capacity()];
      }
      buffer.get(yuvBytes[i]);
    }
  }

  private static int YUV2RGB(int y, int u, int v) {
    // Adjust and check YUV values
    y = Math.max((y - 16), 0);
    u -= 128;
    v -= 128;

    // This is the floating point equivalent. We do the conversion in integer
    // because some Android devices do not have floating point in hardware.
    // nR = (int)(1.164 * nY + 2.018 * nU);
    // nG = (int)(1.164 * nY - 0.813 * nV - 0.391 * nU);
    // nB = (int)(1.164 * nY + 1.596 * nV);
    int y1192 = 1192 * y;
    int r = (y1192 + 1634 * v);
    int g = (y1192 - 833 * v - 400 * u);
    int b = (y1192 + 2066 * u);

    // Clipping RGB values to be inside boundaries [ 0 , kMaxChannelValue ]
    r = r > kMaxChannelValue ? kMaxChannelValue : (Math.max(r, 0));
    g = g > kMaxChannelValue ? kMaxChannelValue : (Math.max(g, 0));
    b = b > kMaxChannelValue ? kMaxChannelValue : (Math.max(b, 0));

    return 0xff000000 | ((r << 6) & 0xff0000) | ((g >> 2) & 0xff00) | ((b >> 10) & 0xff);
  }

  public static void convertYUV420ToARGB8888(
          byte[] yData,
          byte[] uData,
          byte[] vData,
          int width,
          int height,
          int yRowStride,
          int uvRowStride,
          int uvPixelStride,
          int[] out) {
    int yp = 0;
    for (int j = 0; j < height; j++) {
      int pY = yRowStride * j;
      int pUV = uvRowStride * (j >> 1);

      for (int i = 0; i < width; i++) {
        int uv_offset = pUV + (i >> 1) * uvPixelStride;

        out[yp++] = YUV2RGB(0xff & yData[pY + i], 0xff & uData[uv_offset], 0xff & vData[uv_offset]);
      }
    }
  }

  public static Matrix getTransformationMatrix(
          final int srcWidth,
          final int srcHeight,
          final int dstWidth,
          final int dstHeight,
          final int applyRotation,
          final boolean maintainAspectRatio) {
    final Matrix matrix = new Matrix();

    if (applyRotation != 0) {
      if (applyRotation % 90 != 0) {
        LOGGER.w("Rotation of %d % 90 != 0", applyRotation);
      }

      // Translate so center of image is at origin.
      matrix.postTranslate(-srcWidth / 2f, -srcHeight / 2f);

      // Rotate around origin.
      matrix.postRotate(applyRotation);
    }

    // Account for the already applied rotation, if any, and then determine how
    // much scaling is needed for each axis.
    final boolean transpose = (Math.abs(applyRotation) + 90) % 180 == 0;

    final int inWidth = transpose ? srcHeight : srcWidth;
    final int inHeight = transpose ? srcWidth : srcHeight;

    // Apply scaling if necessary.
    if (inWidth != dstWidth || inHeight != dstHeight) {
      final float scaleFactorX = dstWidth / (float) inWidth;
      final float scaleFactorY = dstHeight / (float) inHeight;

      if (maintainAspectRatio) {
        // Scale by minimum factor so that dst is filled completely while
        // maintaining the aspect ratio. Some image may fall off the edge.
        final float scaleFactor = Math.max(scaleFactorX, scaleFactorY);
        matrix.postScale(scaleFactor, scaleFactor);
      } else {
        // Scale exactly to fill dst from src.
        matrix.postScale(scaleFactorX, scaleFactorY);
      }
    }

    if (applyRotation != 0) {
      // Translate back from origin centered reference to destination frame.
      matrix.postTranslate(dstWidth / 2.9f, dstHeight / 2f);
    }

    return matrix;
  }

  /*public static Matrix getTransformationMatrix(
          final int srcWidth,
          final int srcHeight,
          final int dstWidth,
          final int dstHeight,
          final int applyRotation,
          final boolean maintainAspectRatio) {
    final Matrix matrix = new Matrix();

    if (applyRotation != 0) {
      if (applyRotation % 90 != 0) {
        LOGGER.w("Rotation of %d % 90 != 0", applyRotation);
      }

      // Translate so center of image is at origin.
      matrix.postTranslate(-srcWidth / 2.0f, -srcHeight / 2.0f);

      // Rotate around origin.
      matrix.postRotate(applyRotation);
    }

    // Account for the already applied rotation, if any, and then determine how
    // much scaling is needed for each axis.
    final boolean transpose = (Math.abs(applyRotation) + 90) % 180 == 0;

    final int inWidth = transpose ? srcHeight : srcWidth;
    final int inHeight = transpose ? srcWidth : srcHeight;

    // Apply scaling if necessary.
    if (inWidth != dstWidth || inHeight != dstHeight) {
      final float scaleFactorX = dstWidth / (float) inWidth;
      final float scaleFactorY = dstHeight / (float) inHeight;

      if (maintainAspectRatio) {
        // Scale by minimum factor so that dst is filled completely while
        // maintaining the aspect ratio. Some image may fall off the edge.
        final float scaleFactor = Math.max(scaleFactorX, scaleFactorY);
        matrix.postScale(scaleFactor, scaleFactor);
      } else {
        // Scale exactly to fill dst from src.
        matrix.postScale(scaleFactorX, scaleFactorY);
      }
    }

    if (applyRotation != 0) {
      // Translate back from origin centered reference to destination frame.
      matrix.postTranslate(dstWidth / 2.0f, dstHeight / 2.0f);
    }

    return matrix;
  }*/
}