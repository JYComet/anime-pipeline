"""
Text normalization for ASR comparison.
Exact replication of the normalization logic from asr_pipeline.py.
"""
import re
import unicodedata

# ---------------------------------------------------------------------------
# MFA-style punctuation symbols
# ---------------------------------------------------------------------------
_MFA_PUNCT_SYMBOLS = set("〜～")  # wave dash (U+301C), fullwidth tilde (U+FF5E)

# ---------------------------------------------------------------------------
# Chinese numeral -> digit mapping
# ---------------------------------------------------------------------------
_CN_DIGIT_MAP = {
    '零': 0, '〇': 0,
    '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
    '六': 6, '七': 7, '八': 8, '九': 9,
    '两': 2,
}
_CN_NUMERAL_CHARS = set(_CN_DIGIT_MAP.keys()) | {'十', '百', '千', '万', '亿'}

# Fullwidth digits ０-９ -> halfwidth 0-9
_FW_DIGIT_MAP = {chr(0xFF10 + i): str(i) for i in range(10)}

# ---------------------------------------------------------------------------
# Simplified -> Traditional character mapping (~300 common pairs)
# ---------------------------------------------------------------------------
_S2T_MAP: dict[str, str] = {}


def _build_s2t_map():
    if _S2T_MAP:
        return _S2T_MAP
    groups = """
        说説 来來 国國 时時 对對 会會 过過 个個 们們
        为爲爲 学學 开開 关關 门門 车車 长長 见見 贝貝
        页頁 风風 飞飛 马馬 鱼魚 鸟鳥 龙龍 东東 爱愛
        笔筆 变變 边邊 宾賓 仓倉 产產 尝嘗 厂廠 处處
        从從 达達 带帶 当當噹 党黨 导導 灯燈 敌敵 点點
        电電 动動 独獨 断斷 队隊 尔爾 范範 丰豐 妇婦
        刚剛 纲綱 给給 广廣 规規 汉漢 号號 后後 华華
        欢歡 还還 击擊 极極 几幾 计計 济濟 价價 坚堅
        间間 检檢 简簡 剑劍 节節 紧緊 进進 惊驚 旧舊
        举舉 剧劇 据據 决決 军軍 蓝藍 乐樂 离離 礼禮
        连連 练練 两兩 疗療 邻鄰 灵靈 领領 刘劉 录錄
        绿綠 论論 妈媽 买買 卖賣 满满滿 没沒 梦夢 难難
        脑腦 宁寧 农農 暖暖暖 盘盤 钱錢 强強 轻轻輕 热熱
        认認 伤傷 声声聲 师師 实實 识識 势勢 试試 书書
        术術 树樹 双雙 岁歲 孙孫 体體 条條 铁鐵 听聽
        头頭 图圖 万万萬 网網 问問 无無 线線 乡鄉 写寫
        谢謝 兴興 选選 压壓 严嚴 颜顏 业業 医醫 艺藝
        阴陰 应應 拥擁 优優 邮郵 圆圆圓 运運 杂雜 战戰
        张張 阵陣 证證 织織 职職 质質 众眾 转轉 装裝
        资資 总總 组組 讲講 误誤 员員 显顯 调調 议議
        谈談 读讀 诗詩 词詞 语語 课課 谁誰 让讓 记記
        话話 请請 胜勝 卫衛 洁潔 显顯 响響 预預 页頁
        发髮發 回迴 汇匯彙 尽儘盡 历歷曆 台臺颱檯
        复復複 团團糰 脏脏髒臟 云雲 制製 面麵麪
        里裡裏 准準 后後 只隻 征徵 系係繫 钟鐘鍾
    """.split()
    for group in groups:
        s = group[0]
        for t in group[1:]:
            _S2T_MAP[t] = s
    _S2T_MAP["喫"] = "吃"
    _S2T_MAP["鎗"] = "枪"
    _S2T_MAP["兇"] = "凶"
    _S2T_MAP["採"] = "采"
    _S2T_MAP["綵"] = "彩"
    _S2T_MAP["瀋"] = "沈"
    _S2T_MAP["誌"] = "志"
    _S2T_MAP["慾"] = "欲"
    return _S2T_MAP


# ---------------------------------------------------------------------------
# CJK character variant mapping
# ---------------------------------------------------------------------------
_CJK_VARIANT_MAP: dict[str, str] = {}


def _build_cjk_variant_map():
    if _CJK_VARIANT_MAP:
        return _CJK_VARIANT_MAP
    variants = {
        "爲": "為", "峯": "峰", "羣": "群", "峽": "峡",
        "麪": "面", "祕": "秘", "薑": "姜", "禦": "御",
        "綫": "线", "跡": "迹", "蹟": "迹",
        "弐": "二", "壱": "一", "弌": "一", "拾": "十",
    }
    _CJK_VARIANT_MAP.update(variants)
    return _CJK_VARIANT_MAP


# ---------------------------------------------------------------------------
# Chinese number parsing
# ---------------------------------------------------------------------------
def _parse_cn_number(s: str) -> int:
    """Parse a Chinese numeral string (e.g. 十二, 一百二十三) to integer."""
    if not s:
        return 0
    if all(ch in _CN_DIGIT_MAP for ch in s):
        return int("".join(str(_CN_DIGIT_MAP[ch]) for ch in s))

    total = 0
    cur_num = 0
    cur_section = 0

    for ch in s:
        if ch in _CN_DIGIT_MAP:
            cur_num = _CN_DIGIT_MAP[ch]
        elif ch == '十':
            cur_section += cur_num * 10 if cur_num else 10
            cur_num = 0
        elif ch == '百':
            cur_section += cur_num * 100 if cur_num else 100
            cur_num = 0
        elif ch == '千':
            cur_section += cur_num * 1000 if cur_num else 1000
            cur_num = 0
        elif ch == '万':
            cur_section = (cur_section + cur_num) * 10000 if (cur_section + cur_num) > 0 else 10000
            total += cur_section
            cur_section = 0
            cur_num = 0
        elif ch == '亿':
            cur_section = (cur_section + cur_num) * 100000000 if (cur_section + cur_num) > 0 else 100000000
            total += cur_section
            cur_section = 0
            cur_num = 0

    return total + cur_section + cur_num


def _normalize_chinese_numbers(text: str) -> str:
    """Convert Chinese numerals and fullwidth digits to Arabic numerals."""
    result = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in _FW_DIGIT_MAP:
            result.append(_FW_DIGIT_MAP[ch])
            i += 1
        elif ch in _CN_NUMERAL_CHARS:
            j = i
            while j < len(text) and text[j] in _CN_NUMERAL_CHARS:
                j += 1
            cn_str = text[i:j]
            try:
                result.append(str(_parse_cn_number(cn_str)))
            except Exception:
                result.append(cn_str)
            i = j
        else:
            result.append(ch)
            i += 1
    return "".join(result)


# ---------------------------------------------------------------------------
# Public normalization functions
# ---------------------------------------------------------------------------
def normalize_text_mfa(text: str) -> str:
    """Normalize text for ASR comparison (MFA-style, used for segment comparison).

    Steps:
      1. Strip Unicode punctuation / whitespace + wave dash / fullwidth tilde
      2. Unify third-person pronouns (他/她/它/祂/牠 -> 他)
      3. Normalize Chinese numerals and fullwidth digits to Arabic digits
    """
    # 1. Strip punctuation and whitespace
    cleaned = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("P") or cat.startswith("Z") or ch in _MFA_PUNCT_SYMBOLS:
            continue
        cleaned.append(ch)
    text = "".join(cleaned)

    # 2. Unify third-person pronouns
    text = re.sub(r'[他她它祂牠]', '他', text)

    # 3. Normalize Chinese numbers to Arabic digits
    text = _normalize_chinese_numbers(text)

    return text


def normalize_text(text: str) -> str:
    """Normalize text for comparison — deep normalization.

    Steps applied in order:
      1. NFKC — fullwidth/halfwidth, ligatures, compatibility chars
      2. Simplified-Traditional mapping
      3. CJK variant characters
      4. Japanese katakana -> hiragana
      5. Third-person pronouns (他/她/它/祂/牠 -> 他)
      6. Common ASR homophone pairs
      7. Remove spaces between CJK characters
      8. Strip punctuation — keep only letters, numbers, and spaces
      9. Lowercase
     10. Collapse whitespace
    """
    # 1. NFKC normalization
    text = unicodedata.normalize("NFKC", text)

    # 2. Simplified-Traditional mapping
    s2t = _build_s2t_map()
    text = "".join(s2t.get(ch, ch) for ch in text)

    # 3. CJK variant characters
    cjk_var = _build_cjk_variant_map()
    text = "".join(cjk_var.get(ch, ch) for ch in text)

    # 4. Japanese katakana -> hiragana (Unicode block shift: 0x60)
    def _kana_shift(ch):
        cp = ord(ch)
        if 0x30A1 <= cp <= 0x30F6:
            return chr(cp - 0x60)
        if 0x30F7 <= cp <= 0x30FA:
            return chr(0x3041 + (cp - 0x30F7))
        if cp == 0x30FB:  # katakana middle dot
            return " "
        if cp == 0x30FC:  # long vowel mark
            return ""
        return ch
    text = "".join(_kana_shift(ch) for ch in text)

    # 5. Third-person pronouns
    text = re.sub(r'[他她它祂牠]', '他', text)

    # 6. Common ASR homophone pairs
    text = re.sub(r'[得地]', '的', text)
    text = re.sub(r'再', '在', text)
    text = re.sub(r'作', '做', text)
    text = re.sub(r'[吗嘛]', '么', text)

    # 7. Remove spaces between CJK characters
    _CJK = re.compile(
        r'([一-鿿㐀-䶿豈-﫿぀-ゟ゠-ヿ가-힯])'
        r'\s+'
        r'(?=[一-鿿㐀-䶿豈-﫿぀-ゟ゠-ヿ가-힯])'
    )
    text = _CJK.sub(r'\1', text)

    # 8. Strip punctuation — keep only letters, numbers, and spaces
    cleaned = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("N") or cat == "Zs":
            cleaned.append(ch)
        else:
            cleaned.append(" ")
    text = "".join(cleaned)

    # 9 & 10. Lowercase and collapse whitespace
    return " ".join(text.split()).lower()
