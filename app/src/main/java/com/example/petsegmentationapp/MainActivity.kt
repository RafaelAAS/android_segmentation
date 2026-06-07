package com.example.petsegmentationapp

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {

    private var interpreter: Interpreter? = null
    private lateinit var imageViewOriginal: ImageView
    private lateinit var imageViewMask: ImageView
    private var selectedBitmap: Bitmap? = null

    // Tamanho esperado pelo modelo
    private val IMG_SIZE = 128

    // Launcher para abrir a galeria
    private val pickImageLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let {
            val inputStream = contentResolver.openInputStream(it)
            selectedBitmap = BitmapFactory.decodeStream(inputStream)

            // Coloca a foto intocada na tela de CIMA
            imageViewOriginal.setImageBitmap(selectedBitmap)
            imageViewMask.setImageDrawable(null) // Limpa a tela de baixo
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val btnSelectImage = findViewById<Button>(R.id.btnSelectImage)
        val btnSegment = findViewById<Button>(R.id.btnSegment)
        imageViewOriginal = findViewById(R.id.imageViewOriginal)
        imageViewMask = findViewById(R.id.imageViewMask)

        setupModel()

        btnSelectImage.setOnClickListener {
            pickImageLauncher.launch("image/*")
        }

        btnSegment.setOnClickListener {
            if (selectedBitmap != null && interpreter != null) {
                runInference(selectedBitmap!!)
            } else {
                Toast.makeText(this, "Selecione uma imagem ou aguarde o modelo carregar.", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun setupModel() {
        try {
            val fileDescriptor = assets.openFd("model.tflite")
            val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = fileDescriptor.startOffset
            val declaredLength = fileDescriptor.declaredLength
            val tfliteModel = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)

            interpreter = Interpreter(tfliteModel)
        } catch (e: Exception) {
            Toast.makeText(this, "Erro ao carregar modelo: ${e.message}", Toast.LENGTH_LONG).show()
            e.printStackTrace()
        }
    }

    private fun runInference(bitmap: Bitmap) {
        val inputShape = interpreter?.getInputTensor(0)?.shape() ?: return
        val isNHWC = inputShape.size == 4 && inputShape[3] == 3

        // 1. Pré-processamento
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, IMG_SIZE, IMG_SIZE, true)
        val inputBuffer = ByteBuffer.allocateDirect(1 * 3 * IMG_SIZE * IMG_SIZE * 4)
        inputBuffer.order(ByteOrder.nativeOrder())

        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)
        val intValues = IntArray(IMG_SIZE * IMG_SIZE)
        resizedBitmap.getPixels(intValues, 0, resizedBitmap.width, 0, 0, resizedBitmap.width, resizedBitmap.height)

        if (isNHWC) {
            for (i in 0 until IMG_SIZE) {
                for (j in 0 until IMG_SIZE) {
                    val pixelValue = intValues[i * IMG_SIZE + j]
                    inputBuffer.putFloat(((pixelValue shr 16 and 0xFF) / 255.0f - mean[0]) / std[0])
                    inputBuffer.putFloat(((pixelValue shr 8 and 0xFF) / 255.0f - mean[1]) / std[1])
                    inputBuffer.putFloat(((pixelValue and 0xFF) / 255.0f - mean[2]) / std[2])
                }
            }
        } else {
            for (c in 0..2) {
                for (i in 0 until IMG_SIZE) {
                    for (j in 0 until IMG_SIZE) {
                        val pixelValue = intValues[i * IMG_SIZE + j]
                        val channelVal = when (c) {
                            0 -> ((pixelValue shr 16 and 0xFF) / 255.0f - mean[0]) / std[0]
                            1 -> ((pixelValue shr 8 and 0xFF) / 255.0f - mean[1]) / std[1]
                            else -> ((pixelValue and 0xFF) / 255.0f - mean[2]) / std[2]
                        }
                        inputBuffer.putFloat(channelVal)
                    }
                }
            }
        }
        inputBuffer.rewind()

        // 2. Inferência
        val outputShape = interpreter?.getOutputTensor(0)?.shape() ?: return
        val outIsNHWC = outputShape.size == 4 && outputShape[3] == 3

        val outputBuffer = ByteBuffer.allocateDirect(1 * 3 * IMG_SIZE * IMG_SIZE * 4)
        outputBuffer.order(ByteOrder.nativeOrder())

        try {
            interpreter?.run(inputBuffer, outputBuffer)
        } catch (e: Exception) {
            Toast.makeText(this, "Erro ao executar IA: ${e.message}", Toast.LENGTH_LONG).show()
            e.printStackTrace()
            return
        }

        outputBuffer.rewind()

        val outFloatBuffer = outputBuffer.asFloatBuffer()
        val outArray = FloatArray(3 * IMG_SIZE * IMG_SIZE)
        outFloatBuffer.get(outArray)

        // 3. Pós-processamento (Criando a máscara base)
        val maskBitmap = Bitmap.createBitmap(IMG_SIZE, IMG_SIZE, Bitmap.Config.ARGB_8888)

        for (y in 0 until IMG_SIZE) {
            for (x in 0 until IMG_SIZE) {
                val probClass0: Float
                val probClass1: Float
                val probClass2: Float

                if (outIsNHWC) {
                    val baseIndex = (y * IMG_SIZE + x) * 3
                    probClass0 = outArray[baseIndex + 0]
                    probClass1 = outArray[baseIndex + 1]
                    probClass2 = outArray[baseIndex + 2]
                } else {
                    val planeSize = IMG_SIZE * IMG_SIZE
                    val pixelIndex = y * IMG_SIZE + x
                    probClass0 = outArray[0 * planeSize + pixelIndex]
                    probClass1 = outArray[1 * planeSize + pixelIndex]
                    probClass2 = outArray[2 * planeSize + pixelIndex]
                }

                var maxProb = probClass0
                var predictedClass = 0

                if (probClass1 > maxProb) { maxProb = probClass1; predictedClass = 1 }
                if (probClass2 > maxProb) { predictedClass = 2 }

                val color = when (predictedClass) {
                    0 -> Color.GREEN        // Animal
                    1 -> Color.TRANSPARENT  // Fundo Totalmente Invisível
                    2 -> Color.YELLOW       // Borda
                    else -> Color.TRANSPARENT
                }
                maskBitmap.setPixel(x, y, color)
            }
        }

        // 4. Mágica do Layout: Junta a foto original com a máscara translúcida
        val overlayBitmap = createOverlayBitmap(bitmap, maskBitmap)

        // Joga a foto com a máscara sobreposta na tela de BAIXO
        imageViewMask.setImageBitmap(overlayBitmap)
    }

    private fun createOverlayBitmap(originalBitmap: Bitmap, maskBitmap: Bitmap): Bitmap {
        // Cria uma tela em branco com as mesmas dimensões da foto original
        val finalBitmap = Bitmap.createBitmap(originalBitmap.width, originalBitmap.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(finalBitmap)
        val paint = Paint()

        // 1. Pinta a foto original no fundo
        canvas.drawBitmap(originalBitmap, 0f, 0f, paint)

        // 2. Estica a máscara (que tem 128x128) para o tamanho da foto original
        val scaledMask = Bitmap.createScaledBitmap(maskBitmap, originalBitmap.width, originalBitmap.height, true)

        // 3. Aplica 50% de transparência na tinta (0 a 255)
        paint.alpha = 128

        // 4. "Carimba" a máscara translúcida por cima da foto original
        canvas.drawBitmap(scaledMask, 0f, 0f, paint)

        return finalBitmap
    }

    override fun onDestroy() {
        super.onDestroy()
        interpreter?.close()
    }
}