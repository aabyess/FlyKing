using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

/// <summary>
/// 초파리 뇌 판단 서버(factory/server/brain_server.py)와 TCP 127.0.0.1:8790, 줄 단위 JSON으로 이야기한다.
/// 판단은 전부 서버의 진짜 뇌 계산이다. 이 클래스는 요청을 보내고 답을 메인 스레드로 넘겨줄 뿐이다.
/// </summary>
public class BrainClient : MonoBehaviour
{
    public string host = "127.0.0.1";
    public int port = 8790;

    public event Action<Dictionary<string, object>> OnMessage;
    public bool Connected => client != null && client.Connected;
    public bool Ready => Connected && WorkersReady > 0;
    public int WorkersReady { get; private set; }
    public int Workers { get; private set; }
    public int Queue { get; private set; }

    TcpClient client;
    NetworkStream stream;
    readonly ConcurrentQueue<Dictionary<string, object>> inbox = new ConcurrentQueue<Dictionary<string, object>>();
    float retryAt, statusAt;

    void Update()
    {
        if (!Connected && Time.unscaledTime >= retryAt)
        {
            retryAt = Time.unscaledTime + 2f;
            TryConnect();
        }
        if (Connected && Time.unscaledTime >= statusAt)
        {
            statusAt = Time.unscaledTime + 1f;
            Send(new Dictionary<string, object> { ["type"] = "status" });
        }
        while (inbox.TryDequeue(out var msg))
        {
            string type = MiniJson.Text(msg, "type");
            if (type == "hello" || type == "status")
            {
                WorkersReady = (int)MiniJson.Num(msg, "workers_ready");
                Workers = (int)MiniJson.Num(msg, "workers");
                Queue = (int)MiniJson.Num(msg, "queue");
            }
            OnMessage?.Invoke(msg);
        }
    }

    void TryConnect()
    {
        try
        {
            client = new TcpClient();
            client.Connect(host, port);
            stream = client.GetStream();
            new Thread(ReadLoop) { IsBackground = true }.Start();
        }
        catch (Exception)
        {
            client = null;
            WorkersReady = 0;
        }
    }

    void ReadLoop()
    {
        try
        {
            var reader = new StreamReader(stream, new UTF8Encoding(false));
            string line;
            while ((line = reader.ReadLine()) != null)
                if (MiniJson.Parse(line) is Dictionary<string, object> d) inbox.Enqueue(d);
        }
        catch (Exception) { }
    }

    public void Send(Dictionary<string, object> msg)
    {
        if (!Connected) return;
        byte[] bytes = Encoding.UTF8.GetBytes(MiniJson.Serialize(msg) + "\n");
        try { stream.Write(bytes, 0, bytes.Length); }
        catch (Exception) { client = null; WorkersReady = 0; }
    }

    void OnDestroy()
    {
        try { client?.Close(); } catch (Exception) { }
    }
}
