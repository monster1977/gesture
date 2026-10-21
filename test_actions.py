import sys

sys.path.insert(0, r"C:\i\gesture")

import gesture_app as ga


class Recorder:
    def __init__(self):
        self.calls = []

    def keybd_event(self, vk, scan, flags, extra):
        self.calls.append((vk, flags))


class MockAudio:
    def __init__(self):
        self.calls = []

    def mute_all(self):
        self.calls.append("mute")
        return True

    def unmute_all(self):
        self.calls.append("unmute")
        return True

    def change_volume_all(self, delta):
        self.calls.append(("vol", round(delta, 3)))
        return True


assert ga.APP_VERSION

for spec, _label in ga.COMMAND_CATALOG:
    assert ga.parse_action(spec) is not None, spec
for spec in ga.DEFAULT_BINDINGS.values():
    assert ga.parse_action(spec) is not None, spec

assert ga.parse_action("key:0xB0") == ("key", 0xB0)
assert ga.parse_action("key:179") == ("key", 179)
assert ga.parse_action("combo:0x44") == ("combo", (0x44,))
assert ga.parse_action("audio:mute_all") == ("audio", "mute_all")
assert ga.parse_action("bogus") is None

assert ga.match_binding({"A": "key:1"}, []) is None
assert ga.match_binding({"A": "key:1"}, ["A"]) == "key:1"
assert ga.match_binding({"A": "key:1", "B": "key:2"}, ["X", "B"]) == "key:2"

kc = ga.KeyController()
kc.user32 = Recorder()
audio = MockAudio()

assert ga.execute_action(("key", 0xB3), audio, kc) is True
assert kc.user32.calls == [(0xB3, 1), (0xB3, 3)], kc.user32.calls
kc.user32.calls.clear()

assert ga.execute_action(("combo", (0x44,)), audio, kc) is True
assert kc.user32.calls == [(0x5B, 0), (0x44, 0), (0x44, 2), (0x5B, 2)]
kc.user32.calls.clear()

assert ga.execute_action(("audio", "mute_all"), audio, kc) is True
assert ga.execute_action(("audio", "unmute_all"), audio, kc) is True
assert ga.execute_action(("audio", "vol_up"), audio, kc, 0.05) is True
assert ga.execute_action(("audio", "vol_down"), audio, kc, 0.05) is True
assert audio.calls == ["mute", "unmute", ("vol", 0.05), ("vol", -0.05)]

assert ga.execute_action(("audio", "mute_all"), None, kc) is False

labels = set(ga.ALL_GESTURES)
for gesture in ga.RESERVED_GESTURES:
    assert gesture in labels, gesture
for gesture in ga.DEFAULT_BINDINGS:
    assert gesture in labels, gesture
assert not (set(ga.RESERVED_GESTURES) & set(ga.DEFAULT_BINDINGS))

print("ALL ACTION TESTS PASSED")
