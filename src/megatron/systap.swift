// System-audio capture via Core Audio process tap (macOS 14.4+).
// Emits one ASCII header line "SAMPLERATE CHANNELS\n", then raw interleaved
// Float32 PCM to stdout until killed. No loopback driver, no output switching.
import AudioToolbox
import CoreAudio
import Foundation

func check(_ status: OSStatus, _ what: String) {
    guard status == noErr else {
        FileHandle.standardError.write("systap: \(what) failed (\(status))\n".data(using: .utf8)!)
        exit(1)
    }
}

// Tap everything sent to the system output (exclude no processes).
let tapDesc = CATapDescription(stereoGlobalTapButExcludeProcesses: [])
tapDesc.name = "megatron-tap"
tapDesc.isPrivate = true
tapDesc.muteBehavior = .unmuted
var tapID = AudioObjectID(kAudioObjectUnknown)
check(AudioHardwareCreateProcessTap(tapDesc, &tapID), "create process tap (check System Audio Recording permission)")

var asbd = AudioStreamBasicDescription()
var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
var addr = AudioObjectPropertyAddress(
    mSelector: kAudioTapPropertyFormat,
    mScope: kAudioObjectPropertyScopeGlobal,
    mElement: kAudioObjectPropertyElementMain)
check(AudioObjectGetPropertyData(tapID, &addr, 0, nil, &size, &asbd), "get tap format")

// Anchor the aggregate to the current default output device for clocking.
var outDev = AudioDeviceID(0)
size = UInt32(MemoryLayout<AudioDeviceID>.size)
addr.mSelector = kAudioHardwarePropertyDefaultOutputDevice
check(AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, &outDev),
      "get default output")
var outUID: CFString = "" as CFString
size = UInt32(MemoryLayout<CFString>.size)
addr.mSelector = kAudioDevicePropertyDeviceUID
check(AudioObjectGetPropertyData(outDev, &addr, 0, nil, &size, &outUID), "get output UID")

// ponytail: anchored to the output device at launch; switching outputs mid-run
// stops capture — restart megatron after switching.
let aggDesc: [String: Any] = [
    kAudioAggregateDeviceNameKey as String: "megatron-agg",
    kAudioAggregateDeviceUIDKey as String: UUID().uuidString,
    kAudioAggregateDeviceIsPrivateKey as String: true,
    kAudioAggregateDeviceTapAutoStartKey as String: true,
    kAudioAggregateDeviceSubDeviceListKey as String: [[kAudioSubDeviceUIDKey as String: outUID]],
    kAudioAggregateDeviceTapListKey as String: [[
        kAudioSubTapDriftCompensationKey as String: true,
        kAudioSubTapUIDKey as String: tapDesc.uuid.uuidString,
    ]],
]
var aggID = AudioObjectID(kAudioObjectUnknown)
check(AudioHardwareCreateAggregateDevice(aggDesc as CFDictionary, &aggID), "create aggregate device")

let out = FileHandle.standardOutput
out.write("\(Int(asbd.mSampleRate)) \(asbd.mChannelsPerFrame)\n".data(using: .utf8)!)

var procID: AudioDeviceIOProcID?
check(AudioDeviceCreateIOProcIDWithBlock(&procID, aggID, nil) { _, inData, _, _, _ in
    let abl = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: inData))
    for buf in abl where buf.mData != nil && buf.mDataByteSize > 0 {
        out.write(Data(bytes: buf.mData!, count: Int(buf.mDataByteSize)))
    }
}, "create IO proc")
check(AudioDeviceStart(aggID, procID), "start capture")
dispatchMain()  // taps/aggregates are torn down by CoreAudio when we die
