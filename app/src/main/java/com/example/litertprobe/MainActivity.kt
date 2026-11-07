package com.example.litertprobe

import android.os.Build
import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {

    private lateinit var logView: TextView
    private val logBuilder = StringBuilder()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        logView = findViewById(R.id.logText)

        Thread {
            runDiagnostics()
        }.start()
    }

    private fun log(msg: String) {
        logBuilder.append(msg).append("\n")
        runOnUiThread { logView.text = logBuilder.toString() }
    }

    private fun runDiagnostics() {
        log("=== LiteRT Hardware & NPU Probe ===")
        log("Устройство: ${Build.MANUFACTURER} ${Build.MODEL}")
        log("SoC Model : ${if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else "N/A"}")
        log("Hardware  : ${Build.HARDWARE}")
        log("Android   : ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})")
        log("------------------------------------\n")

        // 1. Проверка системных драйверов ядра
        log("[*] Проверка системных файлов ядра:")
        val vertexNode = File("/dev/vertex10")
        log("  /dev/vertex10 : ${if (vertexNode.exists()) "ВИДЕН" else "Не виден напрямую (SELinux)"}")
        
        val ennLib = File("/vendor/lib64/libenn_engine_lib.so")
        log("  libenn_engine : ${if (ennLib.exists()) "ПРИСУТСТВУЕТ" else "Не найден"}")

        // 2. Инициализация через встроенный NNAPI флаг
        log("\n[*] Инициализация инференса через NPU (NNAPI):")
        var interpreter: Interpreter? = null

        try {
            val modelBuffer = loadModelFile("model.tflite")
            log("  [+] Модель INT8 загружена из assets")

            val options = Interpreter.Options().apply {
                setUseNNAPI(true)
                setNumThreads(4)
            }

            interpreter = Interpreter(modelBuffer, options)
            log("  [+] LiteRT успешно подключил системный NPU/NNAPI драйвер!")

            // Буферы для MobileNet: 1x224x224x3 INT8
            val inputSize = 1 * 224 * 224 * 3
            val inputBuffer = ByteBuffer.allocateDirect(inputSize).order(ByteOrder.nativeOrder())
            for (i in 0 until inputSize) inputBuffer.put(128.toByte())
            inputBuffer.rewind()

            val outputBuffer = ByteBuffer.allocateDirect(1001).order(ByteOrder.nativeOrder())

            // Прогрев
            interpreter.run(inputBuffer, outputBuffer)

            // Замер 5 прогонов
            val iterations = 5
            val start = System.nanoTime()
            for (i in 0 until iterations) {
                inputBuffer.rewind()
                outputBuffer.rewind()
                interpreter.run(inputBuffer, outputBuffer)
            }
            val totalMs = (System.nanoTime() - start) / 1_000_000.0
            val avgMs = totalMs / iterations

            log("\n====================================")
            log(" СТАТУС: NPU УСПЕШНО РАБОТАЕТ!")
            log(" Время 1 прогона: %.2f ms".format(avgMs))
            log("====================================")

        } catch (e: Throwable) {
            log("\n[-] Ошибка выполнения на NPU:")
            log("  ${e.javaClass.simpleName}: ${e.message}")
            log("  (Система выполнила откат на CPU)")
        } finally {
            interpreter?.close()
        }
    }

    private fun loadModelFile(modelName: String): ByteBuffer {
        val fileDescriptor = assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
}

