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
import com.google.ai.edge.litert.Accelerator
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var logView: TextView
    private lateinit var scrollView: ScrollView
    private val logBuilder = StringBuilder()

    // Количество треков для бенчмарка
    private val testTrackCount = 3

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
                runAudioBenchmark()
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
            log("[!] Предоставьте доступ ко всем файлам.")
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
            name.contains("mert", ignoreCase = true) && name.endsWith(".tflite")
        }
        return files?.firstOrNull()
    }

    private fun findAudioFiles(): List<File> {
        val validExts = setOf("mp3", "flac", "wav", "m4a", "ogg")
        val searchDirs = listOf(
            File(Environment.getExternalStorageDirectory(), "Music"),
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
        )

        for (dir in searchDirs) {
            if (!dir.exists()) continue
            val found = dir.walkTopDown()
                .filter { it.isFile && it.extension.lowercase() in validExts }
                .take(testTrackCount)
                .toList()
            if (found.isNotEmpty()) return found
        }
        return emptyList()
    }

    private suspend fun runAudioBenchmark() = withContext(Dispatchers.IO) {
        log("=== БЕНЧМАРК MERT-330M FULL TRACK ===")
        log("SoC: ${if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else Build.HARDWARE}\n")

        val modelFile = findModel()
        if (modelFile == null) {
            log("[-] ОШИБКА: Модель mert*.tflite не найдена в /sdcard/Download/")
            return@withContext
        }
        log("[+] Модель: ${modelFile.name} (${modelFile.length() / (1024 * 1024)} MB)")

        val audioFiles = findAudioFiles()
        if (audioFiles.isEmpty()) {
            log("[-] ОШИБКА: Аудиофайлы (mp3/flac/m4a) не найдены в /sdcard/Music или Download")
            return@withContext
        }
        log("[+] Найдено треков: ${audioFiles.size}\n")

        // ==========================================
        // 1. ТЕСТ НА CPU
        // ==========================================
        log("[*] 1. Инициализация FullTrackProcessor на CPU...")
        var cpuTotalInfMs = 0L
        var cpuTotalChunks = 0

        try {
            val cpuProcessor = FullTrackProcessor(modelFile, Accelerator.CPU)
            log("  [+] Модель загружена на CPU")

            for ((i, file) in audioFiles.withIndex()) {
                log("  [*] [CPU ${i + 1}/${audioFiles.size}] Обработка: ${file.name}...")
                val res = cpuProcessor.processFullTrack(file)
                cpuTotalInfMs += res.inferenceMs
                cpuTotalChunks += res.chunkCount

                log("      Чанков: ${res.chunkCount} | Декод: ${res.decodeMs} ms | Инференс: ${res.inferenceMs} ms")
            }
            cpuProcessor.close()
        } catch (e: Throwable) {
            log("[-] Ошибка CPU: ${e.message}")
        }

        // ==========================================
        // 2. ТЕСТ НА GPU (AMD Xclipse)
        // ==========================================
        log("\n[*] 2. Инициализация FullTrackProcessor на GPU...")
        var gpuTotalInfMs = 0L
        var gpuTotalChunks = 0

        try {
            val gpuProcessor = FullTrackProcessor(modelFile, Accelerator.GPU)
            log("  [+] Шейдеры скомпилированы на GPU")

            for ((i, file) in audioFiles.withIndex()) {
                log("  [*] [GPU ${i + 1}/${audioFiles.size}] Обработка: ${file.name}...")
                val res = gpuProcessor.processFullTrack(file)
                gpuTotalInfMs += res.inferenceMs
                gpuTotalChunks += res.chunkCount

                log("      Чанков: ${res.chunkCount} | Декод: ${res.decodeMs} ms | Инференс: ${res.inferenceMs} ms")
            }
            gpuProcessor.close()

            // ==========================================
            // ИТОГИ
            // ==========================================
            if (cpuTotalChunks > 0 && gpuTotalChunks > 0) {
                val cpuAvgChunkMs = cpuTotalInfMs.toDouble() / cpuTotalChunks
                val gpuAvgChunkMs = gpuTotalInfMs.toDouble() / gpuTotalChunks
                val speedup = cpuAvgChunkMs / gpuAvgChunkMs

                log("\n====================================")
                log(" РЕЗУЛЬТАТЫ СРАВНЕНИЯ ($cpuTotalChunks чанков по 10 сек):")
                log(" CPU среднее на чанк : %.0f ms".format(cpuAvgChunkMs))
                log(" GPU среднее на чанк : %.0f ms".format(gpuAvgChunkMs))
                log(" Разница             : %.2fx ускорение".format(speedup))
                log("------------------------------------")
                val track3MinChunks = 35 // 3 минуты с шагом 5 сек = 35 окон
                val est3MinCpuSec = (cpuAvgChunkMs * track3MinChunks) / 1000.0
                val est3MinGpuSec = (gpuAvgChunkMs * track3MinChunks) / 1000.0
                log(" РАСЧЕТ НА ТРЕК 3 МИНУТЫ:")
                log(" CPU : %.1f сек".format(est3MinCpuSec))
                log(" GPU : %.1f сек".format(est3MinGpuSec))
                log("====================================")
            }
        } catch (e: Throwable) {
            log("[-] Ошибка GPU: ${e.message}")
        }
    }
}

