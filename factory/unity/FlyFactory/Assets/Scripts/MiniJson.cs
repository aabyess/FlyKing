using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

/// <summary>뇌 서버와 주고받는 줄 단위 JSON용 작은 파서·직렬화기(외부 패키지 없이).</summary>
public static class MiniJson
{
    public static object Parse(string s)
    {
        int i = 0;
        return Value(s, ref i);
    }

    static void Ws(string s, ref int i)
    {
        while (i < s.Length && char.IsWhiteSpace(s[i])) i++;
    }

    static bool Word(string s, int i, string w) => i + w.Length <= s.Length && string.CompareOrdinal(s, i, w, 0, w.Length) == 0;

    static object Value(string s, ref int i)
    {
        Ws(s, ref i);
        if (i >= s.Length) return null;
        char c = s[i];
        if (c == '{')
        {
            var d = new Dictionary<string, object>();
            i++;
            Ws(s, ref i);
            if (i < s.Length && s[i] == '}') { i++; return d; }
            while (i < s.Length)
            {
                Ws(s, ref i);
                string k = Str(s, ref i);
                Ws(s, ref i);
                i++; // ':'
                d[k] = Value(s, ref i);
                Ws(s, ref i);
                if (i < s.Length && s[i] == ',') { i++; continue; }
                i++; // '}'
                break;
            }
            return d;
        }
        if (c == '[')
        {
            var l = new List<object>();
            i++;
            Ws(s, ref i);
            if (i < s.Length && s[i] == ']') { i++; return l; }
            while (i < s.Length)
            {
                l.Add(Value(s, ref i));
                Ws(s, ref i);
                if (i < s.Length && s[i] == ',') { i++; continue; }
                i++; // ']'
                break;
            }
            return l;
        }
        if (c == '"') return Str(s, ref i);
        if (Word(s, i, "true")) { i += 4; return true; }
        if (Word(s, i, "false")) { i += 5; return false; }
        if (Word(s, i, "null")) { i += 4; return null; }
        int start = i;
        while (i < s.Length && "+-0123456789.eE".IndexOf(s[i]) >= 0) i++;
        double.TryParse(s.Substring(start, i - start), NumberStyles.Float, CultureInfo.InvariantCulture, out double num);
        return num;
    }

    static string Str(string s, ref int i)
    {
        var sb = new StringBuilder();
        i++; // 여는 따옴표
        while (i < s.Length)
        {
            char c = s[i++];
            if (c == '"') break;
            if (c != '\\') { sb.Append(c); continue; }
            char e = s[i++];
            switch (e)
            {
                case 'n': sb.Append('\n'); break;
                case 't': sb.Append('\t'); break;
                case 'r': sb.Append('\r'); break;
                case 'b': sb.Append('\b'); break;
                case 'f': sb.Append('\f'); break;
                case 'u': sb.Append((char)Convert.ToInt32(s.Substring(i, 4), 16)); i += 4; break;
                default: sb.Append(e); break;
            }
        }
        return sb.ToString();
    }

    public static string Serialize(object o)
    {
        var sb = new StringBuilder();
        Write(sb, o);
        return sb.ToString();
    }

    static void Write(StringBuilder sb, object o)
    {
        switch (o)
        {
            case null: sb.Append("null"); break;
            case string str:
                sb.Append('"');
                foreach (char c in str)
                {
                    if (c == '"') sb.Append("\\\"");
                    else if (c == '\\') sb.Append("\\\\");
                    else if (c == '\n') sb.Append("\\n");
                    else if (c < ' ') sb.Append("\\u").Append(((int)c).ToString("x4"));
                    else sb.Append(c);
                }
                sb.Append('"');
                break;
            case bool b: sb.Append(b ? "true" : "false"); break;
            case IDictionary dict:
                sb.Append('{');
                bool first = true;
                foreach (DictionaryEntry kv in dict)
                {
                    if (!first) sb.Append(',');
                    first = false;
                    Write(sb, kv.Key.ToString());
                    sb.Append(':');
                    Write(sb, kv.Value);
                }
                sb.Append('}');
                break;
            case IList list:
                sb.Append('[');
                for (int k = 0; k < list.Count; k++)
                {
                    if (k > 0) sb.Append(',');
                    Write(sb, list[k]);
                }
                sb.Append(']');
                break;
            case IFormattable f: sb.Append(f.ToString(null, CultureInfo.InvariantCulture)); break;
            default: Write(sb, o.ToString()); break;
        }
    }

    public static Dictionary<string, object> Obj(Dictionary<string, object> d, string key) =>
        d != null && d.TryGetValue(key, out var v) ? v as Dictionary<string, object> : null;

    public static double Num(Dictionary<string, object> d, string key, double fallback = 0) =>
        d != null && d.TryGetValue(key, out var v) && v is double x ? x : fallback;

    public static string Text(Dictionary<string, object> d, string key, string fallback = "") =>
        d != null && d.TryGetValue(key, out var v) && v is string s ? s : fallback;
}
