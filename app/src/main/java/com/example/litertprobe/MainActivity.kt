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
            runCompiledModelAudit()
        }.start()
    }

    private fun log(msg: String) {
        logBuilder.append(msg).append("\n")
        runOnUiThread {
            logView.text = logBuilder.toString()
            scrollView.post { scrollView.fullScroll(ScrollView.FOCUS_DOWN) }
        }
    }

    private fun runCompiledModelAudit() {
        log("=== LITERТ COMPILED MODEL AUDIT ===")
        log("SoC     : ${if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else Build.HARDWARE}")
        log("Система : Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})\n")

        val modelAsset = "model.tflite"
        val inputSize = 1 * 224 * 224 * 3
        val inputData = FloatArray(inputSize) { 0.5f }
        val iterations = 10

        // 1. Тест на CPU
        log("[*] 1. Запуск CompiledModel(Accelerator.CPU)...")
        var cpuAvgMs = 0.0
        try {
            val cpuModel = CompiledModel.create(
                assets,
                modelAsset,
                CompiledModel.Options(Accelerator.CPU)
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

        // 2. Тест на GPU (для аппаратного ориентира графики AMD Xclipse)
        log("\n[*] 2. Запуск CompiledModel(Accelerator.GPU)...")
        var gpuAvgMs = 0.0
        try {
            val gpuModel = CompiledModel.create(
                assets,
                modelAsset,
                CompiledModel.Options(Accelerator.GPU)
            )
            val gpuInputs = gpuModel.createInputBuffers()
            val gpuOutputs = gpuModel.createOutputBuffers()
            gpuInputs[0].writeFloat(inputData)
            gpuModel.run(gpuInputs, gpuOutputs)

            val start = System.nanoTime()
            for (i in 0 until iterations) {
                gpuInputs[0].writeFloat(inputData)
                gpuModel.run(gpuInputs, gpuOutputs)
            }
            gpuAvgMs = ((System.nanoTime() - start) / 1_000_000.0) / iterations
            log("  -> Время на GPU: %.2f ms".format(gpuAvgMs))
            gpuModel.close()
        } catch (e: Throwable) {
            log("[-] Ошибка на GPU: ${e.message}")
        }

        // 3. Тест на NPU
        log("\n[*] 3. Запуск CompiledModel(Accelerator.NPU)...")
        try {
            val npuInitStart = System.currentTimeMillis()
            val npuModel = CompiledModel.create(
                assets,
                modelAsset,
                CompiledModel.Options(Accelerator.NPU)
            )
            val npuInitMs = System.currentTimeMillis() - npuInitStart
            log("  [+] Время создания NPU-модели: ${npuInitMs} ms")

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

            log("\n====================================")
            log(" РЕЗУЛЬТАТЫ СРАВНЕНИЯ:")
            log(" CPU : %.2f ms".format(cpuAvgMs))
            log(" GPU : %.2f ms".format(gpuAvgMs))
            log(" NPU : %.2f ms".format(npuAvgMs))
            log("====================================")

            npuModel.close()
        } catch (e: Throwable) {
            log("\n[-] Ошибка инициализации NPU:")
            log("    ${e.javaClass.simpleName}: ${e.message}")
        }
    }
}

