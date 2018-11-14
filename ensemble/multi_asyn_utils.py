import asyncio


async def sub_main(in_queue, out_queue, processor):
    while not in_queue.empty():
        item = in_queue.get()
        ret = processor(item)
        out_queue.put(ret)


def sub_loop(in_queue, out_queue, processor):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(sub_main(in_queue, out_queue, processor))


async def start(executor, in_queue, out_queue, processor):
    await asyncio.get_event_loop().run_in_executor(executor, sub_loop, in_queue, out_queue, processor)
