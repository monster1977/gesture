import os
import sys
import tempfile

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication

import gesture_qt as gq
import gesture_app as ga

gq.CONFIG_PATH = os.path.join(tempfile.gettempdir(), "rufp_test_settings_config.json")
if os.path.exists(gq.CONFIG_PATH):
    os.remove(gq.CONFIG_PATH)

app = QApplication(sys.argv)

# --- 1. Битые спецификации не роняют разбор ---
assert ga.parse_action("key:") is None
assert ga.parse_action("key:zzz") is None
assert ga.parse_action("key:0x") is None
assert ga.parse_action("combo:0x44,oops") is None
assert ga.parse_action("combo:") is None
assert ga.parse_action("") is None
assert ga.parse_action("unknown:1") is None
assert ga.parse_action("key:0xB3") == ("key", 0xB3)

# --- 2. Жест «большой палец вбок» должен быть привязываемым ---
assert "Большой палец (вбок)" in ga.ALL_GESTURES
assert len(set(ga.ALL_GESTURES)) == len(ga.ALL_GESTURES), "дубли жестов в ALL_GESTURES"

# --- 3. Семантика привязок: {} = «без команд», None = дефолтные ---
dlg_empty = gq.SettingsDialog(dict(gq.DEFAULT_CONFIG), None, [0], {})
assert dlg_empty.bindings == {}, "пустой bindings не должен подменяться дефолтными"

dlg_default = gq.SettingsDialog(dict(gq.DEFAULT_CONFIG), None, [0], None)
assert dlg_default.bindings == ga.DEFAULT_BINDINGS

gq.MainWindow.start_worker = lambda self: None
win = gq.MainWindow(dict(gq.DEFAULT_CONFIG))
assert win.bindings == ga.DEFAULT_BINDINGS, "конфиг без bindings = дефолтные"

cfg = gq.load_config()
cfg["bindings"] = {}
gq.save_config(cfg)
assert gq.load_config()["bindings"] == {}, "пустые привязки должны сохраняться"

# --- 4. Применение настроек не падает и обновляет задержку ---
win.apply_settings(
    {
        "camera": 0,
        "hands": 2,
        "delay": 1.5,
        "volume_step": 20.0,
        "roi_enabled": False,
        "post_url": "",
        "post_enabled": False,
        "actions_enabled": True,
        "bindings": dict(ga.DEFAULT_BINDINGS),
    }
)
assert win.action_gate.delay == 1.5, "задержка ActionGate не обновилась"
assert win.config["delay"] == 1.5
assert win.bindings == ga.DEFAULT_BINDINGS


# --- 5. Своя клавиша: список не разрастается, отмена возвращает прежнее ---
class FakeInputDialog:
    result = ("0xB3", True)

    @staticmethod
    def getText(*args, **kwargs):
        return FakeInputDialog.result


gq.QInputDialog = FakeInputDialog

dlg = gq.SettingsDialog(dict(gq.DEFAULT_CONFIG), None, [0], win.bindings)
gesture = "Открытая ладонь"
row = ga.ALL_GESTURES.index(gesture)
combo = dlg.tbl_bindings.cellWidget(row, 1)
base_count = combo.count()

custom_idx = combo.findData(gq.SettingsDialog.CUSTOM_KEY)
combo.setCurrentIndex(custom_idx)
assert dlg.bindings[gesture] == "key:179", dlg.bindings
assert combo.count() == base_count + 1, "должен появиться один пункт «Своя:»"

# повторный ввод другого кода не должен оставлять старый пункт
FakeInputDialog.result = ("0xB1", True)
combo.setCurrentIndex(combo.findData("key:0xB0"))
combo.setCurrentIndex(custom_idx)
assert dlg.bindings[gesture] == "key:177", dlg.bindings
assert combo.count() == base_count + 1, f"пунктов в списке: {combo.count()}"

# отмена ввода возвращает привязку, активную до открытия диалога
FakeInputDialog.result = ("", False)
combo.setCurrentIndex(combo.findData("key:0xB0"))
combo.setCurrentIndex(custom_idx)
assert dlg.bindings[gesture] == "key:0xB0", dlg.bindings

# отмена не теряет уже заданную «Свою:» привязку
FakeInputDialog.result = ("0xB1", True)
combo.setCurrentIndex(custom_idx)
assert dlg.bindings[gesture] == "key:177", dlg.bindings
FakeInputDialog.result = ("", False)
combo.setCurrentIndex(custom_idx)
assert dlg.bindings[gesture] == "key:177", dlg.bindings
assert combo.count() == base_count + 1, f"пунктов в списке: {combo.count()}"

# выбор пункта каталога чистит динамические пункты
combo.setCurrentIndex(combo.findData("key:0xB3"))
assert combo.count() == base_count, f"пунктов в списке: {combo.count()}"
assert dlg.bindings[gesture] == "key:0xB3"

# «Нет команды» убирает привязку
combo.setCurrentIndex(combo.findData(""))
assert gesture not in dlg.bindings
assert dlg.result_config()["bindings"] == dlg.bindings

print("ALL SETTINGS TESTS PASSED")
