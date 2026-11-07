package com.example.litertprobe

import android.os.Build
import android.os.Bundle
import android.system.Os
import android.widget.ScrollView
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
    private lateinit var scrollView: ScrollView
    private val logBuilder = StringBuilder()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        logView = findViewById(R.id.logText)
        scrollView = findViewById(R.id.scrollView)

        Thread {
            runRigorousHardwareProbe()
        }.start()
    }

    private fun log(msg: String) {
        logBuilder.append(msg).append("\n")
        runOnUiThread {
            logView.text = logBuilder.toString()
            scrollView.post { scrollView.fullScroll(ScrollView.FOCUS_DOWN) }
        }
    }

    private fun checkActiveKernelNpuNode(): String? {
        try {
            val fdDir = File("/proc/self/fd")
            val fds = fdDir.listFiles() ?: return null
            for (fd in fds) {
                try {
                    val target = Os.readlink(fd.absolutePath)
                    if (target.contains("vertex") || target.contains("enn")) {
                        return "${fd.name} -> $target"
                    }
                } catch (_: Exception) {}
            }
        } catch (_: Exception) {}
        return null
    }

    private fun runRigorousHardwareProbe() {
        log("=== АППАРАТНЫЙ АУДИТ NPU EXYNOS ===")
        log("SoC: ${if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else Build.HARDWARE}")
        log("Ядро: ${System.getProperty("os.version")}\n")

        val modelBuffer: ByteBuffer
        try {
            modelBuffer = loadModelFile("model.tflite")
            log("[+] Модель INT8 успешно загружена в память")
        } catch (e: Exception) {
            log("[-] Ошибка чтения модели: ${e.message}")
            return
        }

        // Подготовка тестового тензора (MobileNet INT8: 1x224x224x3)
        val inputSize = 1 * 224 * 224 * 3
        val inputBuffer = ByteBuffer.allocateDirect(inputSize).order(ByteOrder.nativeOrder())
        for (i in 0 until inputSize) inputBuffer.put((i % 256).toByte())
        val outputBuffer = ByteBuffer.allocateDirect(1001).order(ByteOrder.nativeOrder())

        val iterations = 10

        // ==========================================
        // ТЕСТ 1: Принудительный чистый CPU
        // ==========================================
        log("\n[*] 1. Запуск на ЧИСТОМ CPU (4 потока, NNAPI выключен)...")
        val cpuOptions = Interpreter.Options().apply {
            setUseNNAPI(false)
            setNumThreads(4)
        }
        val cpuInterpreter = Interpreter(modelBuffer, cpuOptions)

        // Прогрев CPU
        inputBuffer.rewind(); outputBuffer.rewind()
        cpuInterpreter.run(inputBuffer, outputBuffer)

        val cpuStart = System.nanoTime()
        for (i in 0 until iterations) {
            inputBuffer.rewind(); outputBuffer.rewind()
            cpuInterpreter.run(inputBuffer, outputBuffer)
        }
        val cpuTimeMs = ((System.nanoTime() - cpuStart) / 1_000_000.0) / iterations
        log("  -> Время на CPU: %.2f ms".format(cpuTimeMs))
        cpuInterpreter.close()

        // ==========================================
        // ТЕСТ 2: Аппаратный NPU (Samsung Exynos HAL)
        // ==========================================
        log("\n[*] 2. Запуск на NPU (активация драйвера Exynos LiteCore)...")
        val npuOptions = Interpreter.Options().apply {
            setUseNNAPI(true)
        }

        var npuInterpreter: Interpreter? = null
        try {
            npuInterpreter = Interpreter(modelBuffer, npuOptions)

            // Прогрев NPU и компиляция подграфа
            inputBuffer.rewind(); outputBuffer.rewind()
            npuInterpreter.run(inputBuffer, outputBuffer)

            // Проверка файлового дескриптора ядра Linux
            val activeNode = checkActiveKernelNpuNode()
            if (activeNode != null) {
                log("  [✓] ПОДТВЕРЖДЕНИЕ ЯДРА: Дескриптор открыт: $activeNode")
            } else {
                log("  [?] Дескриптор ядра не обнаружен напрямую (SELinux ограничение)")
            }

            val npuStart = System.nanoTime()
            for (i in 0 until iterations) {
                inputBuffer.rewind(); outputBuffer.rewind()
                npuInterpreter.run(inputBuffer, outputBuffer)
            }
            val npuTimeMs = ((System.nanoTime() - npuStart) / 1_000_000.0) / iterations
            log("  -> Время на NPU: %.2f ms".format(npuTimeMs))

            // ==========================================
            // ИТОГОВЫЙ ВЕРДИКТ
            // ==========================================
            val speedup = cpuTimeMs / npuTimeMs
            log("\n====================================")
            log(" РЕЗУЛЬТАТЫ СРАВНЕНИЯ:")
            log(" CPU (4 ядра) : %.2f ms".format(cpuTimeMs))
            log(" NPU Exynos   : %.2f ms".format(npuTimeMs))
            log(" Ускорение    : %.1fx".format(speedup))
            if (speedup >= 3.0) {
                log(" ВЕРДИКТ: АППАРАТНЫЙ NPU РАБОТАЕТ!")
            } else {
                log(" ВЕРДИКТ: Похоже на fallback на процессор")
            }
            log("====================================")

        } catch (e: Exception) {
            log("[-] Ошибка инициализации NPU: ${e.message}")
        } finally {
            npuInterpreter?.close()
        }
    }

    private fun loadModelFile(modelName: String): ByteBuffer {
        val fileDescriptor = assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            fileDescriptor.startOffset,
            fileDescriptor.declaredLength
        )
    }
}

