using System;
using System.IO;
using System.IO.Ports;
using System.Text;

namespace WT310
{
    internal class Program
    {
        // 读取电子设备数据的命令
        private const string GetData = "NUMeric:NORMal:VALue?\r\n";

        // 串口连接相关的参数
        private const string PortName = "COM3";
        private const int BaudRate = 9600;
        private const int DataBits = 8;
        private const StopBits StopBits = System.IO.Ports.StopBits.One;
        private const Parity Parity = System.IO.Ports.Parity.None;

        // 循环次数和程序延时时间
        private const int LoopCount = 10;
        private const int ThreadSleepTime = 10;

        // 待写入文本文件的路径
        private static readonly string FilePath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.Desktop),
            "output.txt");

        static void Main(string[] args)
        {
            // 使用using语句控制串口连接对象的生命周期
            using (SerialPort wt310 = new SerialPort())
            {
                // 配置串口连接参数
                wt310.PortName = PortName;
                wt310.BaudRate = BaudRate;
                wt310.DataBits = DataBits;
                wt310.StopBits = StopBits;
                wt310.Parity = Parity;
                // 打开串口连接
                wt310.Open();
                // 用于保存所有读取的电子设备数据
                StringBuilder allResult = new StringBuilder();
                // 循环读取电子设备数据
                for (int i = 0; i < LoopCount; i++)
                {
                    // 向电子设备写入GetData命令
                    wt310.Write(GetData);
                    // 延时一段时间等待电子设备返回数据,
                    System.Threading.Thread.Sleep(ThreadSleepTime);
                    // 从电子设备读取数据并输出到控制台
                    string result = wt310.ReadLine();
                    Console.WriteLine(result);
                    // 将每次读取的数据添加到allResult字符串中
                    allResult.Append(result);
                }
                // 将保存在allResult字符串中的数据写入指定的文本文件中
                File.WriteAllText(FilePath, allResult.ToString());
                // 程序延时，以确保数据写入操作已经完成
                System.Threading.Thread.Sleep(5000);
            }
        }
    }
}
