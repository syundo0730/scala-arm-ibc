from trio_serial import AbstractSerialStream

from motors.core.command import Command


class SerialClient:
    def __init__(self, serial_stream: AbstractSerialStream):
        self._serial_stream = serial_stream

    async def command(self, command: Command) -> None:
        await self._serial_stream.send_all(command.command_bytes)
        await self._serial_stream.receive_some()

    async def query(self, command: Command):
        await self._serial_stream.send_all(command.command_bytes)
        res = await self._serial_stream.receive_some()
        if not command.response_parser:
            return None
        return command.response_parser(res)
