"""Read indexed members from standard ZIP/ZIP64 volumes via bounded HTTP ranges.

No archive extraction paths are used as filesystem destinations. Media callers
choose their own validated names and verify member size and CRC32.
"""
import struct
import zlib
import fsspec


class RemoteVolumes:
    def __init__(self,urls,sizes):
        if len(urls)!=len(sizes) or not urls:raise ValueError('Invalid volumes')
        self.urls=list(urls);self.sizes=list(sizes)

    def read(self,disk,offset,length):
        if disk<0 or offset<0 or length<0:raise ValueError('Invalid range')
        chunks=[]
        while length:
            if disk>=len(self.urls):raise ValueError('Range exceeds archive volumes')
            if offset>=self.sizes[disk]:
                offset-=self.sizes[disk];disk+=1;continue
            count=min(length,self.sizes[disk]-offset)
            with fsspec.open(self.urls[disk],block_size=65536) as stream:
                stream.seek(offset);data=stream.read(count)
            if len(data)!=count:raise IOError('Truncated archive range')
            chunks.append(data);length-=count;disk+=1;offset=0
        return b''.join(chunks)

    def index(self):
        last=len(self.urls)-1;amount=min(self.sizes[last],65557)
        tail=self.read(last,self.sizes[last]-amount,amount)
        at=tail.rfind(b'PK\x05\x06')
        if at<0:raise ValueError('ZIP end record missing')
        record=struct.unpack_from('<4sHHHHIIH',tail,at)
        disk_start,count,size,offset=record[2],record[4],record[5],record[6]
        locator_at=at-20
        if locator_at>=0 and tail[locator_at:locator_at+4]==b'PK\x06\x07':
            _,z_disk,z_offset,n_disks=struct.unpack_from('<4sIQI',tail,locator_at)
            if n_disks!=len(self.urls):raise ValueError('Volume count mismatch')
            extended=struct.unpack('<4sQHHIIQQQQ',self.read(z_disk,z_offset,56))
            if extended[0]!=b'PK\x06\x06':raise ValueError('ZIP64 end record missing')
            disk_start,count,size,offset=extended[5],extended[7],extended[8],extended[9]
        if size>100*1024**2:raise ValueError('Unexpectedly large central directory')
        directory=self.read(disk_start,offset,size);position=0;entries=[]
        for _ in range(count):
            fields=struct.unpack_from('<4s6H3I5H2I',directory,position)
            if fields[0]!=b'PK\x01\x02':raise ValueError('Bad central-directory signature')
            flags,method,crc,compressed,uncompressed=fields[3],fields[4],fields[7],fields[8],fields[9]
            name_len,extra_len,comment_len,disk,member_offset=fields[10],fields[11],fields[12],fields[13],fields[16]
            name_bytes=directory[position+46:position+46+name_len]
            name=name_bytes.decode('utf-8' if flags&0x800 else 'cp437')
            extra=directory[position+46+name_len:position+46+name_len+extra_len]
            cursor=0
            while cursor+4<=len(extra):
                tag,size_extra=struct.unpack_from('<HH',extra,cursor);cursor+=4
                payload=extra[cursor:cursor+size_extra];cursor+=size_extra
                if tag==1:
                    pos=0
                    expanded=[]
                    for value in [uncompressed,compressed,member_offset]:
                        if value==0xffffffff:
                            value=struct.unpack_from('<Q',payload,pos)[0];pos+=8
                        expanded.append(value)
                    uncompressed,compressed,member_offset=expanded
                    if disk==0xffff:disk=struct.unpack_from('<I',payload,pos)[0]
            if flags&1:raise ValueError('Encrypted ZIP is not supported')
            entries.append({'name':name,'bytes':uncompressed,'compressed_bytes':compressed,
                            'disk':disk,'offset':member_offset,'compress_type':method,'crc32':crc})
            position+=46+name_len+extra_len+comment_len
        return entries

    def member(self,entry):
        disk,offset=entry['disk'],entry['offset']
        header=struct.unpack('<4s5H3I2H',self.read(disk,offset,30))
        if header[0]!=b'PK\x03\x04':raise ValueError('Invalid local header')
        if header[2]&1:raise ValueError('Encrypted member')
        data_start=offset+30+header[-2]+header[-1]
        data=self.read(disk,data_start,entry['compressed_bytes'])
        if entry['compress_type']==8:data=zlib.decompress(data,-15)
        elif entry['compress_type']!=0:raise ValueError('Unsupported ZIP compression')
        if len(data)!=entry['bytes'] or zlib.crc32(data)!=entry['crc32']:
            raise ValueError('Member size/CRC mismatch')
        return data
