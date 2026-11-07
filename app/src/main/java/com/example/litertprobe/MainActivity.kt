package com.example.litertprobe

import android.os.Build
import android.os.Bundle
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.google.ai.edge.litert.*

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
            runStrictCompiledModelAudit()
        }.start()
    }

    private fun log(msg: String) {
        logBuilder.append(msg).append("\n")
        runOnUiThread {
            logView.text = logBuilder.toString()
            scrollView.post { scrollView.fullScroll(ScrollView.FOCUS_DOWN) }
        }
    }

    private fun runStrictCompiledModelAudit() {
        log("=== LITERТ SAMSUNG NPU AUDIT ===")
        log("SoC     : ${if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else Build.HARDWARE}")
        log("Система : Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})\n")

        val modelAsset = "model.tflite"
        val inputSize = 1 * 224 * 224 * 3
        val inputData = FloatArray(inputSize) { 0.5f }
        val iterations = 10

        // 1. Аппаратная проверка совместимости через Samsung SDK
        try {
            val isSupported = NpuCompatibilityChecker.Samsung.isDeviceSupported()
            log("[*] NpuCompatibilityChecker.Samsung: $isSupported")
        } catch (e: Throwable) {
            log("[!] Чекер совместимости недоступен: ${e.message}")
        }

        // 2. Инициализация Environment с NPU провайдером
        val env: Environment
        try {
            log("[*] Создание Environment(BuiltinNpuAcceleratorProvider)...")
            env = Environment.create(BuiltinNpuAcceleratorProvider(this))
            log("[+] Environment инициализирован с системными NPU драйверами")
        } catch (e: Throwable) {
            log("[-] Ошибка создания Environment: ${e.message}")
            return
        }

        // 3. Тест эталона: чистый CPU
        log("\n[*] 1. Запуск на CPU (через CompiledModel + env)...")
        var cpuAvgMs = 0.0
        try {
            val cpuModel = CompiledModel.create(
                assets,
                modelAsset,
                CompiledModel.Options(Accelerator.CPU),
                env
            )
            val cpuInputs = cpuModel.createInputBuffers()
            val cpuOutputs = cpuModel.createOutputBuffers()
            cpuInputs[0].writeFloat(inputData)
            cpuModel.run(cpuInputs, cpuOutputs)

            val start = System.nanoTime()
            for (i in 0 until iterations) {
                cpuInputs[0].writeFloat(inputData)
                cpuModel.run(cpuInputs, cpuOutputs)
            }
            cpuAvgMs = ((System.nanoTime() - start) / 1_000_000.0) / iterations
            log("  -> Время на CPU: %.2f ms".format(cpuAvgMs))
            cpuModel.close()
        } catch (e: Throwable) {
            log("[-] Ошибка на CPU: ${e.message}")
        }

        // 4. Тест аппаратного NPU
        log("\n[*] 2. Запуск на NPU (Samsung Exynos LiteCore + env)...")
        try {
            val npuInitStart = System.currentTimeMillis()
            val npuModel = CompiledModel.create(
                assets,
                modelAsset,
                CompiledModel.Options(Accelerator.NPU),
                env
            )
            val npuInitMs = System.currentTimeMillis() - npuInitStart
            log("  [✓] On-device компиляция графа NPU завершена за ${npuInitMs} ms")

            val npuInputs = npuModel.createInputBuffers()
            val npuOutputs = npuModel.createOutputBuffers()
            npuInputs[0].writeFloat(inputData)
            npuModel.run(npuInputs, npuOutputs)

            val start = System.nanoTime()
            for (i in 0 until iterations) {
                npuInputs[0].writeFloat(inputData)
                npuModel.run(npuInputs, npuOutputs)
            }
            val npuAvgMs = ((System.nanoTime() - start) / 1_000_000.0) / iterations
            log("  -> Время на NPU: %.2f ms".format(npuAvgMs))

            val speedup = if (cpuAvgMs > 0) cpuAvgMs / npuAvgMs else 0.0
            log("\n====================================")
            log(" ИТОГОВЫЙ СТАТУС:")
            log(" CPU время : %.2f ms".format(cpuAvgMs))
            log(" NPU время : %.2f ms".format(npuAvgMs))
            log(" Разница   : %.1fx ускорение".format(speedup))
            if (speedup >= 1.5) {
                log(" ВЕРДИКТ   : АППАРАТНЫЙ NPU EXYNOS РЕАЛЬНО ЗАДЕЙСТВОВАН!")
            } else {
                log(" ВЕРДИКТ   : Ускорения нет (проверьте логи JIT-компилятора)")
            }
            log("====================================")

            npuModel.close()
        } catch (e: Throwable) {
            log("\n[-] Ошибка выполнения на NPU: ${e.javaClass.simpleName}")
            log("    Детали: ${e.message}")
        }
    }
}

