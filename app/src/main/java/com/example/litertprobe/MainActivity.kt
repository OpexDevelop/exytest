package com.example.litertprobe

import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.Settings
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.EmbeddingEngine
import com.google.ai.edge.litertlm.EmbeddingEngineConfig
import com.google.ai.edge.litertlm.InputData
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var logView: TextView
    private lateinit var scrollView: ScrollView
    private val logBuilder = StringBuilder()

    // Количество фото для тестового прогона (экстраполяция покажет время на 1000)
    private val testBatchSize = 30

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        logView = findViewById(R.id.logText)
        scrollView = findViewById(R.id.scrollView)
    }

    override fun onResume() {
        super.onResume()
        if (checkPermissions()) {
            lifecycleScope.launch {
                runBenchmarkPipeline()
            }
        }
    }

    private fun log(msg: String) {
        logBuilder.append(msg).append("\n")
        runOnUiThread {
            logView.text = logBuilder.toString()
            scrollView.post { scrollView.fullScroll(ScrollView.FOCUS_DOWN) }
        }
    }

    private fun checkPermissions(): Boolean {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R && !Environment.isExternalStorageManager()) {
            log("[!] Предоставьте доступ ко всем файлам для чтения фото и модели.")
            try {
                val intent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION).apply {
                    data = Uri.parse("package:$packageName")
                }
                startActivity(intent)
            } catch (_: Exception) {
                startActivity(Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION))
            }
            return false
        }
        return true
    }

    private fun findModel(): File? {
        val downloadDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
        val files = downloadDir.listFiles { _, name ->
            name.contains("embeddinggemma", ignoreCase = true) && name.endsWith(".litertlm")
        }
        return files?.firstOrNull()
    }

    private fun findPhotos(): List<File> {
        val dcimDir = File(Environment.getExternalStorageDirectory(), "DCIM/Camera")
        if (!dcimDir.exists()) return emptyList()

        val validExts = setOf("jpg", "jpeg", "png", "webp")
        return dcimDir.listFiles { file ->
            file.isFile && file.extension.lowercase() in validExts
        }?.take(testBatchSize)?.toList() ?: emptyList()
    }

    private suspend fun runBenchmarkPipeline() = withContext(Dispatchers.IO) {
        log("=== БЕНЧМАРК EMBEDDING GEMMA 2 ===")
        log("SoC: ${if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else Build.HARDWARE}\n")

        val modelFile = findModel()
        if (modelFile == null) {
            log("[-] ОШИБКА: Файл модели *.litertlm не найден в /sdcard/Download/")
            return@withContext
        }
        log("[+] Модель: ${modelFile.name} (${modelFile.length() / (1024 * 1024)} MB)")

        val photoFiles = findPhotos()
        if (photoFiles.isEmpty()) {
            log("[-] ОШИБКА: Фотографии в DCIM/Camera не найдены")
            return@withContext
        }
        log("[+] Найдено ${photoFiles.size} фото для теста (потоковая обработка по 1 фото)\n")

        // ==========================================
        // 1. ПРОГОН НА ЧИСТОМ CPU
        // ==========================================
        log("[*] 1. Инициализация EmbeddingEngine на CPU...")
        var cpuInferenceNs = 0L
        var cpuSuccessCount = 0

        try {
            val cpuConfig = EmbeddingEngineConfig(
                modelPath = modelFile.absolutePath,
                backend = Backend.CPU(),
                visionBackend = Backend.CPU()
            )
            val cpuEngine = EmbeddingEngine(cpuConfig)
            val initStart = System.currentTimeMillis()
            cpuEngine.initialize()
            log("  [+] Инициализация CPU заняла ${System.currentTimeMillis() - initStart} ms")

            log("  [*] Обработка фотографий на CPU...")
            for ((index, file) in photoFiles.withIndex()) {
                val bytes = file.readBytes()
                val input = listOf(InputData.Image(bytes))

                val t0 = System.nanoTime()
                val output = cpuEngine.computeEmbedding(input)
                cpuInferenceNs += (System.nanoTime() - t0)
                cpuSuccessCount++

                if (index == 0) {
                    log("  [✓] Длина выходного вектора: ${output.embedding.size} float")
                }
            }

            val cpuTotalSec = cpuInferenceNs / 1_000_000_000.0
            val cpuFps = cpuSuccessCount / cpuTotalSec
            log("  -> Время чистого инференса CPU: %.2f сек (%.2f фото/сек)".format(cpuTotalSec, cpuFps))
            cpuEngine.close()
        } catch (e: Throwable) {
            log("[-] Сбой CPU: ${e.message}")
        }

        // ==========================================
        // 2. ПРОГОН НА GPU (AMD Xclipse)
        // ==========================================
        log("\n[*] 2. Инициализация EmbeddingEngine на GPU...")
        var gpuInferenceNs = 0L
        var gpuSuccessCount = 0

        try {
            val gpuConfig = EmbeddingEngineConfig(
                modelPath = modelFile.absolutePath,
                backend = Backend.GPU(),
                visionBackend = Backend.GPU()
            )
            val gpuEngine = EmbeddingEngine(gpuConfig)
            val initStart = System.currentTimeMillis()
            gpuEngine.initialize()
            log("  [+] Компиляция GPU пайплайна заняла ${System.currentTimeMillis() - initStart} ms")

            log("  [*] Обработка фотографий на GPU...")
            for (file in photoFiles) {
                val bytes = file.readBytes()
                val input = listOf(InputData.Image(bytes))

                val t0 = System.nanoTime()
                gpuEngine.computeEmbedding(input)
                gpuInferenceNs += (System.nanoTime() - t0)
                gpuSuccessCount++
            }

            val gpuTotalSec = gpuInferenceNs / 1_000_000_000.0
            val gpuFps = gpuSuccessCount / gpuTotalSec
            log("  -> Время чистого инференса GPU: %.2f сек (%.2f фото/сек)".format(gpuTotalSec, gpuFps))
            gpuEngine.close()

            // ==========================================
            // ИТОГОВОЕ СРАВНЕНИЕ
            // ==========================================
            if (cpuSuccessCount > 0 && gpuSuccessCount > 0) {
                val cpuSec = cpuInferenceNs / 1_000_000_000.0
                val gpuSec = gpuInferenceNs / 1_000_000_000.0
                val speedup = cpuSec / gpuSec

                val est1000CpuSec = (cpuSec / cpuSuccessCount) * 1000.0
                val est1000GpuSec = (gpuSec / gpuSuccessCount) * 1000.0

                log("\n====================================")
                log(" РЕЗУЛЬТАТЫ СРАВНЕНИЯ ($testBatchSize фото):")
                log(" Чистый CPU : %.2f сек".format(cpuSec))
                log(" Чистый GPU : %.2f сек".format(gpuSec))
                log(" Разница    : %.2fx ускорение".format(speedup))
                log("------------------------------------")
                log(" ПРОГНОЗ НА 1000 ФОТО:")
                log(" CPU : %.1f сек (%.1f мин)".format(est1000CpuSec, est1000CpuSec / 60))
                log(" GPU : %.1f сек (%.1f мин)".format(est1000GpuSec, est1000GpuSec / 60))
                log("====================================")
            }
        } catch (e: Throwable) {
            log("[-] Сбой GPU: ${e.message}")
        }
    }
}

