import asyncio, httpx, sys
sys.path.insert(0, '.')
from app.core.track_processor import create_default_processor


async def main():
    async with httpx.AsyncClient() as client:
        r = await client.get('http://127.0.0.1:8080/walkbg/static/kml/wutaishan.kml')
        content = r.text
    proc = create_default_processor()
    pts = proc.parse_kml_or_gpx(content, 'kml')
    print(f'解析到轨迹点: {len(pts)} 个')
    if pts:
        print(f'第一个点: lat={pts[0].latitude}, lon={pts[0].longitude}, elev={pts[0].elevation}')
    else:
        # 检查 gx:coord 是否存在
        from xml.etree import ElementTree as ET
        ns = {"kml": "http://www.opengis.net/kml/2.2", "gx": "http://www.google.com/kml/ext/2.2"}
        root = ET.fromstring(content)
        gx_coords = root.findall('.//gx:coord', ns)
        print(f'gx:coord 元素数量: {len(gx_coords)}')
        if gx_coords:
            print(f'第一个 gx:coord: {gx_coords[0].text}')


if __name__ == "__main__":
    asyncio.run(main())
