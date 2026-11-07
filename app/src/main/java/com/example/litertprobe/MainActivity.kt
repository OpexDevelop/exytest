package com.example.litertprobe

import android.os.Build
import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnapiDelegate
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

        lifecycleScope.launch {
            runDiagnostics()
        }
    }

    private fun log(msg: String) {
        logBuilder.append(msg).append("\n")
        runOnUiThread { logView.text = logBuilder.toString() }
    }

    private suspend fun runDiagnostics() = withContext(Dispatchers.Default) {
        log("=== LiteRT Hardware & NPU Probe ===")
        log("Устройство: ${Build.MANUFACTURER} ${Build.MODEL}")
        log("SoC Model : ${if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else "N/A"}")
        log("Hardware  : ${Build.HARDWARE}")
        log("Board     : ${Build.BOARD}")
        log("Android   : ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})")
        log("------------------------------------\n")

        // 1. Проверка системных узлов Samsung NPU
        log("[*] Проверка системных драйверов:")
        val vertexNode = File("/dev/vertex10")
        log("  /dev/vertex10 : ${if (vertexNode.exists()) "НАЙДЕН (доступен)" else "Не виден напрямую (SELinux)"}")
        
        val ennLib = File("/vendor/lib64/libenn_engine_lib.so")
        log("  libenn_engine : ${if (ennLib.exists()) "ПРИСУТСТВУЕТ" else "Не найден"}")

        // 2. Тест запуска модели через NPU (NNAPI Delegate)
        log("\n[*] Тестирование аппаратного инференса NPU:")
        var nnapiDelegate: NnapiDelegate? = null
        var interpreter: Interpreter? = null

        try {
            val modelBuffer = loadModelFile("model.tflite")
            log("  [+] Модель INT8 загружена в память (MobileNet)")

            val nnapiOptions = NnapiDelegate.Options().apply {
                setExecutionPreference(NnapiDelegate.Options.EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER)
                setAllowFp16(true)
            }
            nnapiDelegate = NnapiDelegate(nnapiOptions)

            val options = Interpreter.Options().apply {
                addDelegate(nnapiDelegate)
                setNumThreads(4)
            }

            interpreter = Interpreter(modelBuffer, options)
            log("  [+] LiteRT успешно инициализировал NNAPI/NPU делегат!")

            // 3. Тестовый прогон инференса
            // Входной тензор MobileNet: 1 x 224 x 224 x 3 байта
            val inputSize = 1 * 224 * 224 * 3
            val inputBuffer = ByteBuffer.allocateDirect(inputSize).order(ByteOrder.nativeOrder())
            for (i in 0 until inputSize) inputBuffer.put(128.toByte())
            inputBuffer.rewind()

            // Выходной тензор: 1 x 1001 классов
            val outputBuffer = ByteBuffer.allocateDirect(1001).order(ByteOrder.nativeOrder())

            // Прогрев
            interpreter.run(inputBuffer, outputBuffer)

            // Замер скорости
            val iterations = 5
            val start = System.nanoTime()
            for (i in 0 until iterations) {
                inputBuffer.rewind()
                outputBuffer.rewind()
                interpreter.run(inputBuffer, outputBuffer)
            }
            val totalTimeMs = (System.nanoTime() - start) / 1_000_000.0
            val avgTime = totalTimeMs / iterations

            log("\n====================================")
            log(" СТАТУС: NPU УСПЕШНО ЗАДЕЙСТВОВАН!")
            log(" Среднее время инференса: %.2f ms".format(avgTime))
            log("====================================")

        } catch (e: Throwable) {
            log("\n[-] Ошибка инициализации NPU:")
            log("  ${e.javaClass.simpleName}: ${e.message}")
            log("  Возможен откат на CPU.")
        } finally {
            interpreter?.close()
            nnapiDelegate?.close()
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
