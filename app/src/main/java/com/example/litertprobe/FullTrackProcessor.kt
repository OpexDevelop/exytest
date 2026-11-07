package com.example.litertprobe

import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.File
import java.nio.ByteOrder
import kotlin.math.sqrt

class FullTrackProcessor(modelFile: File, accelerator: Accelerator) {

    private val chunkSize = 240_000
    private val targetSampleRate = 24_000
    private val embeddingDim = 1024

    private val compiledModel: CompiledModel = CompiledModel.create(
        modelFile.absolutePath,
        CompiledModel.Options(accelerator)
    )

    data class TrackResult(
        val embedding: FloatArray,
        val chunkCount: Int,
        val decodeMs: Long,
        val inferenceMs: Long
    )

    fun processFullTrack(audioFile: File): TrackResult {
        val tDecodeStart = System.currentTimeMillis()
        val pcmFloats = decodeToPcm(audioFile)
        val decodeMs = System.currentTimeMillis() - tDecodeStart

        if (pcmFloats.isEmpty()) {
            return TrackResult(FloatArray(embeddingDim), 0, decodeMs, 0L)
        }

        if (pcmFloats.size < chunkSize) {
            val padded = FloatArray(chunkSize)
            System.arraycopy(pcmFloats, 0, padded, 0, pcmFloats.size)

            val tInfStart = System.currentTimeMillis()
            val singleVector = runInference(padded)
            normalize(singleVector)
            val infMs = System.currentTimeMillis() - tInfStart

            return TrackResult(singleVector, 1, decodeMs, infMs)
        }

        val step = chunkSize / 2
        val accumulatedVector = FloatArray(embeddingDim)
        var chunkCount = 0
        val chunkBuffer = FloatArray(chunkSize)
        var offset = 0
        var totalInfMs = 0L

        while (offset + chunkSize <= pcmFloats.size) {
            System.arraycopy(pcmFloats, offset, chunkBuffer, 0, chunkSize)

            val t0 = System.currentTimeMillis()
            val chunkVector = runInference(chunkBuffer)
            totalInfMs += (System.currentTimeMillis() - t0)

            for (i in 0 until embeddingDim) {
                accumulatedVector[i] += chunkVector[i]
            }
            chunkCount++
            offset += step
        }

        if (chunkCount > 0) {
            for (i in 0 until embeddingDim) {
                accumulatedVector[i] /= chunkCount.toFloat()
            }
        }

        normalize(accumulatedVector)

        return TrackResult(accumulatedVector, chunkCount, decodeMs, totalInfMs)
    }

    private fun runInference(inputData: FloatArray): FloatArray {
        val inputs = compiledModel.createInputBuffers()
        val outputs = compiledModel.createOutputBuffers()

        inputs[0].writeFloat(inputData)
        compiledModel.run(inputs, outputs)

        val result = FloatArray(embeddingDim)
        outputs[0].readFloat(result)
        return result
    }

    private fun normalize(vector: FloatArray) {
        var sumSquares = 0.0f
        for (v in vector) sumSquares += v * v
        val norm = sqrt(sumSquares)
        if (norm > 1e-6f) {
            for (i in vector.indices) vector[i] /= norm
        }
    }

    private fun decodeToPcm(file: File): FloatArray {
        val extractor = MediaExtractor()
        try {
            extractor.setDataSource(file.absolutePath)
        } catch (_: Exception) {
            return FloatArray(0)
        }

        var audioTrackIndex = -1
        var format: MediaFormat? = null

        for (i in 0 until extractor.trackCount) {
            val trackFormat = extractor.getTrackFormat(i)
            val mime = trackFormat.getString(MediaFormat.KEY_MIME) ?: ""
            if (mime.startsWith("audio/")) {
                audioTrackIndex = i
                format = trackFormat
                break
            }
        }

        if (audioTrackIndex == -1 || format == null) {
            extractor.release()
            return FloatArray(0)
        }

        extractor.selectTrack(audioTrackIndex)
        val mime = format.getString(MediaFormat.KEY_MIME)!!
        val sampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        val channelCount = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)

        val codec = MediaCodec.createDecoderByType(mime)
        codec.configure(format, null, null, 0)
        codec.start()

        val pcmList = ArrayList<Float>()
        val info = MediaCodec.BufferInfo()
        var isEOS = false

        while (!isEOS) {
            val inIndex = codec.dequeueInputBuffer(10000)
            if (inIndex >= 0) {
                val inBuffer = codec.getInputBuffer(inIndex)!!
                val sampleSize = extractor.readSampleData(inBuffer, 0)
                if (sampleSize < 0) {
                    codec.queueInputBuffer(inIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                    isEOS = true
                } else {
                    codec.queueInputBuffer(inIndex, 0, sampleSize, extractor.sampleTime, 0)
                    extractor.advance()
                }
            }

            var outIndex = codec.dequeueOutputBuffer(info, 10000)
            while (outIndex >= 0) {
                val outBuffer = codec.getOutputBuffer(outIndex)!!
                outBuffer.order(ByteOrder.LITTLE_ENDIAN)

                while (outBuffer.remaining() >= 2 * channelCount) {
                    var sampleSum = 0f
                    for (ch in 0 until channelCount) {
                        sampleSum += (outBuffer.short / 32768.0f)
                    }
                    pcmList.add(sampleSum / channelCount)
                }

                codec.releaseOutputBuffer(outIndex, false)
                outIndex = codec.dequeueOutputBuffer(info, 0)
            }
        }

        codec.stop()
        codec.release()
        extractor.release()

        return resample(pcmList, sampleRate, targetSampleRate)
    }

    private fun resample(source: List<Float>, srcRate: Int, dstRate: Int): FloatArray {
        if (srcRate == dstRate) return source.toFloatArray()
        val ratio = srcRate.toDouble() / dstRate.toDouble()
        val newSize = (source.size / ratio).toInt()
        val result = FloatArray(newSize)

        for (i in 0 until newSize) {
            val srcIndex = (i * ratio).toInt()
            if (srcIndex < source.size) {
                result[i] = source[srcIndex]
            }
        }
        return result
    }

    fun close() {
        compiledModel.close()
    }
}

