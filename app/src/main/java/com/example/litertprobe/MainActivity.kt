package com.example.litertprobe

import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.Settings
import android.widget.Button
import android.widget.EditText
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var logView: TextView
    private lateinit var scrollView: ScrollView
    private lateinit var inputPrompt: EditText
    private lateinit var btnSend: Button

    private val logBuilder = StringBuilder()
    private val lmExecutor = Executors.newSingleThreadExecutor()
    private val lmDispatcher = lmExecutor.asCoroutineDispatcher()

    private var engine: Engine? = null
    private var conversation: Conversation? = null
    private var isInitializing = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        logView = findViewById(R.id.logText)
        scrollView = findViewById(R.id.scrollView)
        inputPrompt = findViewById(R.id.inputPrompt)
        btnSend = findViewById(R.id.btnSend)

        btnSend.setOnClickListener {
            val text = inputPrompt.text.toString().trim()
            if (text.isNotEmpty()) {
                inputPrompt.setText("")
                sendUserMessage(text)
            }
        }
    }

    override fun onResume() {
        super.onResume()
        checkPermissionsAndStart()
    }

    private fun log(msg: String) {
        logBuilder.append(msg).append("\n")
        runOnUiThread {
            logView.text = logBuilder.toString()
            scrollView.post { scrollView.fullScroll(ScrollView.FOCUS_DOWN) }
        }
    }

    private fun appendStream(chunk: String) {
        logBuilder.append(chunk)
        runOnUiThread {
            logView.text = logBuilder.toString()
            scrollView.post { scrollView.fullScroll(ScrollView.FOCUS_DOWN) }
        }
    }

    private fun checkPermissionsAndStart() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R && !Environment.isExternalStorageManager()) {
            log("[!] Требуется разрешение на чтение файла 2.6 ГБ из Загрузок.")
            log("[*] Открываю настройки... Переключите тумблер 'Разрешить доступ' и вернитесь.")
            try {
                val intent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION).apply {
                    data = Uri.parse("package:$packageName")
                }
                startActivity(intent)
            } catch (e: Exception) {
                startActivity(Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION))
            }
            return
        }

        if (engine == null && !isInitializing) {
            lifecycleScope.launch {
                initModelPipeline()
            }
        }
    }

    private fun findModelFile(): File? {
        val downloadDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
        val defaultFile = File(downloadDir, "gemma-4-E2B-it_NPU_Uncompiled.litertlm")
        if (defaultFile.exists() && defaultFile.isFile) return defaultFile

        // Поиск любого другого .litertlm файла в папке Загрузки
        val files = downloadDir.listFiles { _, name -> name.endsWith(".litertlm") }
        if (!files.isNullOrEmpty()) return files[0]

        return null
    }

    private suspend fun initModelPipeline() = withContext(lmDispatcher) {
        isInitializing = true
        log("=== LiteRT-LM Exynos NPU Launcher ===")
        log("Чипсет: ${if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else Build.HARDWARE}")

        val modelFile = findModelFile()
        if (modelFile == null) {
            log("\n[-] Файл модели не найден в папке Загрузки!")
            log("    Ожидался: /sdcard/Download/gemma-4-E2B-it_NPU_Uncompiled.litertlm")
            isInitializing = false
            return@withContext
        }

        log("[+] Найден файл: ${modelFile.name} (${modelFile.length() / (1024 * 1024)} MB)")
        log("[*] Запуск компиляции и загрузки графа в NPU...")
        log("    (Внимание: первый старт On-Device NPU занимает 1-2 минуты!)")

        val startTime = System.currentTimeMillis()

        try {
            // Подключение нативного NPU бэкенда LiteRT-LM
            val config = EngineConfig(
                modelPath = modelFile.absolutePath,
                backend = Backend.NPU(nativeLibraryDir = applicationInfo.nativeLibraryDir),
                cacheDir = cacheDir.path
            )

            val newEngine = Engine(config)
            newEngine.initialize()

            engine = newEngine
            conversation = newEngine.createConversation()

            val loadSec = (System.currentTimeMillis() - startTime) / 1000.0
            log("\n[✓] NPU успешно инициализирован за ${loadSec}s!")
            log("----------------------------------------")

            runOnUiThread { btnSend.isEnabled = true }

            // Стартовый проверочный запрос
            sendUserMessage("Say hello and confirm you are running on NPU in one short sentence.")

        } catch (e: Throwable) {
            log("\n[-] Ошибка загрузки NPU бэкенда: ${e.message}")
            log("    Попытка аварийного отката на GPU...")
            tryFallbackToGpu(modelFile)
        } finally {
            isInitializing = false
        }
    }

    private suspend fun tryFallbackToGpu(modelFile: File) {
        try {
            val gpuConfig = EngineConfig(
                modelPath = modelFile.absolutePath,
                backend = Backend.GPU(),
                cacheDir = cacheDir.path
            )
            val newEngine = Engine(gpuConfig)
            newEngine.initialize()
            engine = newEngine
            conversation = newEngine.createConversation()
            log("[✓] GPU fallback успешен!")
            runOnUiThread { btnSend.isEnabled = true }
        } catch (e: Throwable) {
            log("[-] Ошибка GPU fallback: ${e.message}")
        }
    }

    private fun sendUserMessage(prompt: String) {
        val conv = conversation ?: return
        btnSend.isEnabled = false

        log("\nUser: $prompt")
        appendStream("Gemma 4: ")

        lifecycleScope.launch(lmDispatcher) {
            val start = System.currentTimeMillis()
            conv.sendMessageAsync(prompt)
                .catch { err ->
                    log("\n[-] Ошибка инференса: ${err.message}")
                }
                .collect { chunk ->
                    appendStream(chunk.toString())
                }

            val totalSec = (System.currentTimeMillis() - start) / 1000.0
            log("\n[Генерация завершена за ${totalSec}s]")
            runOnUiThread { btnSend.isEnabled = true }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        lmExecutor.execute {
            conversation?.close()
            engine?.close()
        }
    }
}

