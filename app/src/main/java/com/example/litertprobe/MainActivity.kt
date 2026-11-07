package com.example.litertprobe

import android.os.Build
import android.os.Bundle
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel

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
        log("=== LITERТ COMPILED MODEL API АУДИТ ===")
        log("SoC     : ${if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else Build.HARDWARE}")
        log("Система : Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})\n")

        val modelAsset = "model.tflite"
        val inputSize = 1 * 224 * 224 * 3
        val inputData = FloatArray(inputSize) { 0.5f }
        val iterations = 10

        // ==========================================================
        // 1. Тест эталона: чистый CPU через CompiledModel
        // ==========================================================
        log("[*] 1. Сборка CompiledModel(Accelerator.CPU)...")
        var cpuAvgMs = 0.0
        try {
            val cpuInitStart = System.currentTimeMillis()
            val cpuModel = CompiledModel.create(
                assets,
                modelAsset,
                CompiledModel.Options(Accelerator.CPU)
            )
            val cpuInitMs = System.currentTimeMillis() - cpuInitStart
            log("  [+] CPU граф готов за ${cpuInitMs} ms")

            val cpuInputs = cpuModel.createInputBuffers()
            val cpuOutputs = cpuModel.createOutputBuffers()
            cpuInputs[0].writeFloat(inputData)

            // Прогрев
            cpuModel.run(cpuInputs, cpuOutputs)

            val start = System.nanoTime()
            for (i in 0 until iterations) {
                cpuInputs[0].writeFloat(inputData)
                cpuModel.run(cpuInputs, cpuOutputs)
            }
            cpuAvgMs = ((System.nanoTime() - start) / 1_000_000.0) / iterations
            log("  -> Среднее время на CPU: %.2f ms".format(cpuAvgMs))
            cpuModel.close()
        } catch (e: Throwable) {
            log("[-] Ошибка на CPU: ${e.message}")
        }

        // ==========================================================
        // 2. Тест чистого NPU (СТРОГО Accelerator.NPU, БЕЗ ОТКАТА)
        // ==========================================================
        log("\n[*] 2. Сборка CompiledModel(Accelerator.NPU)...")
        log("    (Откат на CPU/GPU запрещён. Если NPU не отвечает — будет крэш с ошибкой)")

        try {
            val npuInitStart = System.currentTimeMillis()

            // Передаем ТОЛЬКО Accelerator.NPU
            val npuModel = CompiledModel.create(
                assets,
                modelAsset,
                CompiledModel.Options(Accelerator.NPU)
            )
            val npuInitMs = System.currentTimeMillis() - npuInitStart

            log("  [✓] ГРАФ УСПЕШНО СКОМПИЛИРОВАН ПОД EXYNOS LITECORE!")
            log("      Время on-device NPU компиляции: ${npuInitMs} ms")

            val npuInputs = npuModel.createInputBuffers()
            val npuOutputs = npuModel.createOutputBuffers()
            npuInputs[0].writeFloat(inputData)

            // Прогрев
            npuModel.run(npuInputs, npuOutputs)

            val start = System.nanoTime()
            for (i in 0 until iterations) {
                npuInputs[0].writeFloat(inputData)
                npuModel.run(npuInputs, npuOutputs)
            }
            val npuAvgMs = ((System.nanoTime() - start) / 1_000_000.0) / iterations
            log("  -> Среднее время на NPU: %.2f ms".format(npuAvgMs))

            val outputData = npuOutputs[0].readFloat()
            log("  -> Выходной тензор получен (${outputData.size} классов)")

            val speedup = if (cpuAvgMs > 0) cpuAvgMs / npuAvgMs else 0.0
            log("\n====================================")
            log(" ИТОГОВЫЙ СТАТУС:")
            log(" CPU время : %.2f ms".format(cpuAvgMs))
            log(" NPU время : %.2f ms".format(npuAvgMs))
            log(" Разница   : %.1fx ускорение".format(speedup))
            log(" ВЕРДИКТ   : АППАРАТНЫЙ NPU EXYNOS АКТИВЕН!")
            log("====================================")

            npuModel.close()

        } catch (e: Throwable) {
            log("\n[-] ИСКЛЮЧЕНИЕ: NPU отклонил запуск:")
            log("    Класс ошибки: ${e.javaClass.simpleName}")
            log("    Причина     : ${e.message}")
            log("\n[i] Отката на CPU не произошло, так как fallback был выключен.")
        }
    }
}

