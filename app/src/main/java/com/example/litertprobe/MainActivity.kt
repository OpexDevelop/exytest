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

    // Количество фото для бенчмарка
    private val testBatchSize = 50

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
                runEmbeddingBenchmark()
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
            log("[!] Требуется доступ к памяти для чтения фото из DCIM и модели из Download.")
            try {
                val intent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION).apply {
                    data = Uri.parse("package:$packageName")
                }
                startActivity(intent)
            } catch (e: Exception) {
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

        val validExtensions = setOf("jpg", "jpeg", "png", "webp")
        return dcimDir.listFiles { file ->
            file.isFile && file.extension.lowercase() in validExtensions
        }?.take(testBatchSize)?.toList() ?: emptyList()
    }

    private suspend fun runEmbeddingBenchmark() = withContext(Dispatchers.IO) {
        log("=== БЕНЧМАРК EMBEDDING GEMMA 2 ===")
        log("Чипсет: ${if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else Build.HARDWARE}\n")

        val modelFile = findModel()
        if (modelFile == null) {
            log("[-] Файл модели не найден в папке Загрузки!")
            log("    Положи файл *.litertlm (например embeddinggemma-2-740m.litertlm) в /sdcard/Download/")
            return@withContext
        }
        log("[+] Найдена модель: ${modelFile.name} (${modelFile.length() / (1024 * 1024)} MB)")

        val photoFiles = findPhotos()
        if (photoFiles.isEmpty()) {
            log("[-] Фотографии в папке DCIM/Camera не найдены!")
            return@withContext
        }
        log("[*] Загрузка ${photoFiles.size} фото в RAM для исключения дисковых задержек...")
        val photosData = photoFiles.map { it.readBytes() }
        log("[+] Фотографии загружены в память (${photosData.sumOf { it.size } / (1024 * 1024)} MB)\n")

        // ==========================================
        // 1. ТЕСТ НА CPU
        // ==========================================
        log("[*] 1. Инициализация EmbeddingEngine на CPU...")
        var cpuTotalSec = 0.0
        try {
            val cpuConfig = EmbeddingEngineConfig(
                modelPath = modelFile.absolutePath,
                backend = Backend.CPU(),
                visionBackend = Backend.CPU()
            )
            val cpuEngine = EmbeddingEngine(cpuConfig)
            val cpuInitTime = System.currentTimeMillis()
            cpuEngine.initialize()
            log("  [+] CPU движок готов за ${(System.currentTimeMillis() - cpuInitTime)} ms")

            log("  [*] Индексация на CPU...")
            val start = System.currentTimeMillis()
            for ((index, bytes) in photosData.withIndex()) {
                val input = listOf(InputData.Image(bytes))
                val emb = cpuEngine.computeEmbedding(input).embedding
                if (index == 0) {
                    log("  [✓] Размерность вектора: ${emb.size} Float32 чисел")
                }
            }
            cpuTotalSec = (System.currentTimeMillis() - start) / 1000.0
            val cpuFps = photosData.size / cpuTotalSec
            log("  -> CPU время: %.2f сек (%.2f фото/сек)".format(cpuTotalSec, cpuFps))
            cpuEngine.close()
        } catch (e: Throwable) {
            log("[-] Ошибка на CPU: ${e.message}")
        }

        // ==========================================
        // 2. ТЕСТ НА GPU (AMD Xclipse)
        // ==========================================
        log("\n[*] 2. Инициализация EmbeddingEngine на GPU...")
        try {
            val gpuConfig = EmbeddingEngineConfig(
                modelPath = modelFile.absolutePath,
                backend = Backend.GPU(),
                visionBackend = Backend.GPU()
            )
            val gpuEngine = EmbeddingEngine(gpuConfig)
            val gpuInitTime = System.currentTimeMillis()
            gpuEngine.initialize()
            log("  [+] GPU пайплайн готов за ${(System.currentTimeMillis() - gpuInitTime)} ms")

            log("  [*] Индексация на GPU...")
            val start = System.currentTimeMillis()
            for (bytes in photosData) {
                val input = listOf(InputData.Image(bytes))
                gpuEngine.computeEmbedding(input)
            }
            val gpuTotalSec = (System.currentTimeMillis() - start) / 1000.0
            val gpuFps = photosData.size / gpuTotalSec
            log("  -> GPU время: %.2f сек (%.2f фото/сек)".format(gpuTotalSec, gpuFps))
            gpuEngine.close()

            // ==========================================
            // ИТОГИ
            // ==========================================
            val speedup = if (gpuTotalSec > 0) cpuTotalSec / gpuTotalSec else 0.0
            val est1000Cpu = (cpuTotalSec / photosData.size) * 1000.0
            val est1000Gpu = (gpuTotalSec / photosData.size) * 1000.0

            log("\n====================================")
            log(" РЕЗУЛЬТАТЫ СРАВНЕНИЯ (${photosData.size} фото):")
            log(" CPU время : %.2f сек".format(cpuTotalSec))
            log(" GPU время : %.2f сек".format(gpuTotalSec))
            log(" Разница   : %.2fx ускорение".format(speedup))
            log("------------------------------------")
            log(" РАСЧЕТ НА 1000 ФОТО:")
            log(" На CPU : %.1f сек (~%.1f мин)".format(est1000Cpu, est1000Cpu / 60))
            log(" На GPU : %.1f сек (~%.1f мин)".format(est1000Gpu, est1000Gpu / 60))
            log("====================================")

        } catch (e: Throwable) {
            log("[-] Ошибка на GPU: ${e.message}")
        }
    }
}

