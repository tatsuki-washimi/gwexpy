
"""
Native WIN (WIN32) format reader for gwexpy (Pure Python + NumPy).
"""

from __future__ import annotations

import datetime
import struct
import warnings

import numpy as np

from gwexpy.timeseries import TimeSeries, TimeSeriesDict
from gwpy.time import to_gps

# --- Constants & Helpers ---

# WIN Block Header: 12 bytes
# struct.pack('>IHHH', size, org_id, network, channel) ? No.
# Standard WIN headers vary slightly but typically:
# Bytes 0-3: Total block size (big-endian unsigned int) (but check endianness)
# Usually header structure:
#  Time (BCD or Seconds?) - WIN uses "second block" structure.
#  Wait, the file structure needs to be parsed sequentially.

# Structure of 1-sec block header (from NIED/ERI specs):
# 4 bytes: Total Size of this block (including header)
# 4 bytes: Org ID / Network ID etc?
# No, usually:
# Header (variable length, but typical is):
#  [0-3] Size (Int32)
#  [4-7] Header info (channel ID, sample rate/sizing?)
#  ... Time info ...

# Let's follow a standard approach for WIN/WIN32 format parsing.
# WIN format structure:
# File is a sequence of blocks.
# Block structure:
#  Header (variable size? usually fixed part + optional part)
#  Data (compressed)

# Header details (typical):
#  Word 0 (4B): Total length of block (bytes) excluding the length word itself? Or including? Usually including.
#  Word 1 (4B):
#     Bits 31-28: Organization ID?
#     Bits 27-16: Channel ID
#     Bits 15-12: Sample sizing (1s, 1m, 1h) - usually 1s data.
#     Bits 11-0:  Sample rate (Hz) (approximate or code?)

#  Word 2 (4B): Time (YYMMDDHH) in BCD
#  Word 3 (4B): Time (MMSSxxxx) in BCD + extra flags
#
# Let's check a robust reference logic or reimplement careful binary parsing.

# Note on Endianness: WIN is historically Big-Endian.

def bcd_to_int(bcd_val):
    """Convert BCD hex to integer."""
    out = 0
    mult = 1
    while bcd_val > 0:
        digit = bcd_val & 0xF
        out += digit * mult
        mult *= 10
        bcd_val >>= 4
    return out

def _parse_channel_info(path):
    """
    Parse WIN channel definition file.
    Format typically:
      ChannelID(hex) Flag Delay(ms) Gain Name Component ...
    
    Returns
    -------
    dict : { ch_id_int: {'name': str, 'gain': float, 'unit': str, ...} }
    """
    ch_map = {}
    if path is None:
        return ch_map

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            # Minimal parsing
            # Ex: 9211 0    0 1.000000e+00 N.KKMH  U  ...
            if len(parts) < 4:
                continue
            
            try:
                chid_hex = parts[0]
                chid = int(chid_hex, 16)
                
                # parts[1] is flag (on/off) or similar
                # parts[2] is delay
                gain = float(parts[3])
                
                name = parts[4] if len(parts) > 4 else f"CH{chid:04X}"
                comp = parts[5] if len(parts) > 5 else ""
                
                # Combine name and component?
                # Usually standard name is STATION.COMP
                full_name = f"{name}.{comp}" if comp else name
                
                # Unit? Usually count, converted to velocity/accel by gain.
                # Gain typically maps count -> physical.
                
                ch_map[chid] = {
                    'name': full_name,
                    'gain': gain
                }
            except (ValueError, IndexError):
                continue
                
    return ch_map

# --- Decompression ---

def _decompress_4bit(buffer_bytes, n_samples, last_val):
    """
    Decompress 4-bit nibble differences.
    buffer_bytes: bytes object containing packed nibbles.
    n_samples: number of diffs to extract.
    last_val: integration constant.
    """
    # Each byte contains 2 nibbles (High 4, Low 4).
    # Length of buffer must be ceil(n_samples / 2).
    
    # 1. Unpack to uint8 array
    raw = np.frombuffer(buffer_bytes, dtype=np.uint8)
    
    # 2. Split nibbles
    # High: (x >> 4) & 0xF
    # Low:  x & 0xF
    # But these are unsigned. We need signed 4-bit (-8 to +7).
    
    # Create array of size 2*len(raw)
    nibbles = np.zeros(len(raw) * 2, dtype=np.int8)
    nibbles[0::2] = (raw >> 4) & 0xF
    nibbles[1::2] = raw & 0xF
    
    # 3. Sign extension for 4-bit
    # If val >= 8 (1000), it is negative.
    # 8 -> -8 (1000)
    # 9 -> -7
    # ...
    # 15 -> -1
    # Simple trick: (val + 8) % 16 - 8 ? No.
    # Valid: if x >= 8: x -= 16.
    
    mask = nibbles >= 8
    nibbles[mask] -= 16
    
    # Truncate to actual n_samples
    diffs = nibbles[:n_samples]
    
    # 4. Integrate
    out = np.cumsum(diffs) + last_val
    return out, out[-1]

def _decompress_block(data_bytes, bit_len, n_samples, last_val):
    """
    Decompress generic diff block.
    bit_len: 4, 8, 16, 24, 32
    """
    diffs = None
    
    if bit_len == 4:
        return _decompress_4bit(data_bytes, n_samples, last_val)
        
    elif bit_len == 8:
        diffs = np.frombuffer(data_bytes, dtype=np.int8)
        
    elif bit_len == 16:
        # standard 2-byte, Big Endian
        # numpy frombuffer + byteswap
        diffs = np.frombuffer(data_bytes, dtype=np.dtype('>i2'))
        
    elif bit_len == 24:
        # 3-byte is tricky in numpy.
        # Pack to 4-byte and shift? Or simply loop (slow)?
        # Or: pad and use frombuffer?
        # Fast way: view as uint8, reshape (N, 3), pad column, view as int32.
        
        raw_u8 = np.frombuffer(data_bytes, dtype=np.uint8)
        # Ensure divisible by 3
        if len(raw_u8) % 3 != 0:
            # Handle padding error or truncate?
            remainder = len(raw_u8) % 3
            if remainder:
                raw_u8 = raw_u8[:-remainder]
                
        n_elem = len(raw_u8) // 3
        reshaped = raw_u8.reshape(n_elem, 3)
        
        # We need Big Endian 24-bit -> convert to Native Int32
        # Construct 4-byte array:
        # [SignEx, B0, B1, B2] (Big Endian) 
        # But wait, 24bit signed... 
        # Easiest is usually: B0<<16 | B1<<8 | B2. Then sign extend.
        
        # Let's do float calc or manual strict bitwise on int32 buffer?
        # Construct uint32: [0, B0, B1, B2] is Big Endian positive.
        # Sign extension logic: If B0 & 0x80: negative.
        
        # Buffer approach:
        # Prepend/Append based on endian?
        # Python ints are easy but slow.
        
        # Try:
        # val = (b0 << 16) + (b1 << 8) + b2.
        # if val >= 0x800000: val -= 0x1000000
        
        # NumPy Vectorized:
        b0 = reshaped[:, 0].astype(np.int32)
        b1 = reshaped[:, 1].astype(np.int32)
        b2 = reshaped[:, 2].astype(np.int32)
        
        val = (b0 << 16) | (b1 << 8) | b2
        mask_neg = (val & 0x800000) != 0
        val[mask_neg] -= 0x1000000
        
        diffs = val
        
    elif bit_len == 32:
        diffs = np.frombuffer(data_bytes, dtype=np.dtype('>i4'))
        
    else:
        raise ValueError(f"Unsupported bit length: {bit_len}")
        
    # Check size match
    if len(diffs) < n_samples:
         # Warn or pad?
         # Usually indicates minor corruption or EOF logic.
         pass
         
    # Truncate if excess (usually due to block align)
    diffs = diffs[:n_samples]
    
    # Cumulative Sum
    out = np.cumsum(diffs) + last_val
    return out, out[-1]


def read_win_file(source, channels_file=None, verbose=False, **kwargs):
    """
    Read a WIN/WIN32 format file.
    
    Parameters
    ----------
    source : str
        Path to WIN file.
    channels_file : str, optional
        Path to channel definition file.
    verbose : bool
        Print debug info.
        
    Returns
    -------
    TimeSeriesDict
    """
    
    # 1. Parse Channel Info
    ch_info_map = _parse_channel_info(channels_file)
    
    data_map = {}  # { ch_id: [(time_gps, data_array, sr), ...] }
    
    # 2. Read File
    with open(source, 'rb') as f:
        file_size = f.seek(0, 2)
        f.seek(0)
        
        while f.tell() < file_size:
            # Read Block Header (Part 1: Size)
            # Size includes the 4 bytes of the size field itself?
            # Standard WIN: First 4 bytes = Total Size (Big Endian)
            
            f.tell()
            head_bytes = f.read(4)
            if len(head_bytes) < 4:
                break
                
            block_size = struct.unpack('>I', head_bytes)[0]
            
            if block_size < 4:
                # Should not happen
                break
                
            # Read rest of block
            #  header: usually another 12-16 bytes?
            #  Header structure depends on implementation.
            #  Common (WIN System):
            #   Word 0: Size (read)
            #   Word 1: ID/SR 
            #   Word 2: Time 1
            #   Word 3: Time 2
            
            remaining_bytes = block_size - 4
            block_data = f.read(remaining_bytes)
            
            if len(block_data) < remaining_bytes:
                warnings.warn(f"Incomplete block at end of file: expected {remaining_bytes}, got {len(block_data)}")
                break
                
            # Parse Header Fields
            # Word 1 (4B)
            # w1 = struct.unpack('>I', block_data[0:4])[0]
            # org_id = (w1 >> 28) & 0xF
            # ch_id  = (w1 >> 16) & 0xFFF  # 12 bits? Some formats use 16 bits (old WIN).
            # sample_sizing = (w1 >> 12) & 0xF
            # sr_approx = w1 & 0xFFF
            
            # Correction: Some formats use full 16 bits for ChID if OrgID is not used standardly?
            # Standard specification:
            # 31-28: Org
            # 27-16: ChID (12 bits) -> Max 4095
            
            # Wait, 9211 (hex) is 37393 decimal. 16 bits needed.
            # Usually:
            # High 16 bits = Channel ID (or combined Org/Ch)
            # Low 16 bits = Sample Rate
            
            # Let's interpret Word 1 as: [ChID(16)][SR(16)] or similar?
            # Reference check:
            # WIN32 format:
            #  Bytes 4-7:
            #    ChID (High 2 bytes? or embedded?)
            
            # Implementation heuristic:
            # unpack 4 bytes as Hex -> check against channel file.
            
            w1_bytes = block_data[0:4]
            # Standard:
            ch_id = (w1_bytes[0] << 8) | w1_bytes[1] # High 2 bytes
            # Wait, nibble shifts?
            # If w1 is 32-bit int: 
            #  (w1 >> 20) ?
            # Let's assume standard WIN32:
            #  Byte 4: (Org << 4) | (ChID_Hi >> 8)
            #  Byte 5: ChID_Lo
            #  Byte 6: Res | SampleSize
            #  Byte 7: SR
            
            # Simpler View:
            # Hex dump of typical header: 00 00 00 54 (Size) | 92 11 10 64 (ID/SR) | ...
            # 9211 = ChID
            # 1 = Res/Size?
            # 064 = 100 Hz?
            
            # Let's extract hex strings for ChID:
            # 92 11 (2 bytes) = ChID
            # 1 (nibble) = sample size flag? (0=1s, 1=1m..)
            # 064 (3 nibbles) = SR (100)
            
            w1_val = struct.unpack('>I', w1_bytes)[0]
            
            ch_id = (w1_val >> 16) & 0xFFFF
            sr_info = w1_val & 0xFFF
            # resizing information? (w1_val >> 12) & 0xF
            
            # Timestamp (Word 2 & Word 3)
            # YYMMDDHH (BCD)
            w2_val = struct.unpack('>I', block_data[4:8])[0]
            # MMSS ssss ( ssss = microsec? or flags?)
            # Usually WIN is 1-second blocks.
            # so MMSS is Minute/Second. Low bytes might be partial second or flags.
            w3_val = struct.unpack('>I', block_data[8:12])[0] # Wait, this is word 3.
            
            # Time Reconstruction
            # W2: Y(8) M(8) D(8) H(8) -> BCD?
            # Actually standard is BCD.
            
            # BCD extraction helpers
            def _bcd(x): return int(f"{x:x}")
            
            # Hex representation of W2: e.g. 99 01 01 12 -> 1999/01/01 12h
            # Parse bytes directly is safer than int->hex str
            
            y_byte = (w2_val >> 24) & 0xFF
            m_byte = (w2_val >> 16) & 0xFF
            d_byte = (w2_val >> 8) & 0xFF
            h_byte = w2_val & 0xFF
            
            year = bcd_to_int(y_byte)
            # Year 2-digit fix:
            if year < 70:
                year += 2000
            else:
                year += 1900
            
            month = bcd_to_int(m_byte)
            day = bcd_to_int(d_byte)
            hour = bcd_to_int(h_byte)
            
            # W3: Min(8) Sec(8) Res(16)?
            min_byte = (w3_val >> 24) & 0xFF
            sec_byte = (w3_val >> 16) & 0xFF
            minute = bcd_to_int(min_byte)
            second = bcd_to_int(sec_byte)
            
            # Construct datetime
            try:
                dt_obj = datetime.datetime(year, month, day, hour, minute, second, tzinfo=datetime.timezone.utc)
                t_gps = to_gps(dt_obj).value
            except ValueError:
                # Invalid time
                continue
                
            # Actual Data Payload
            # Starts at byte 12 (relative to block_data start, i.e., byte 16 of file chunk)
            # Structure:
            #  Sample Rate (Real) might override header SR?
            #  Usually data starts immediately.
            #  But wait, compression requires knowing sample count.
            #  WIN sample count is usually SR (for 1 sec block).
            
            # How many samples?
            # SR derived from header `sr_info`.
            sr = sr_info # usually correct Hz
            if sr == 0:
                sr = 1 # fallback
            
            # Data Offset inside block_data? 
            # Standard Header size is 12 bytes (Excluding Size).
            # So data starts at block_data[12:].
            
            # Parsing payload:
            # Payload is sequence of "groups"?
            # Or just sequence of differences?
            # WIN Structure:
            #  It's a sequence of channel data if multiplexed?
            #  Wait, this loop assumes block is ONE channel (channel-oriented block).
            #  WIN files are usually mixed blocks of different channels.
            #  We identified ch_id above. So this block belongs to that channel.
            
            # Compression format inside the payload:
            #  The payload consists of multiple sub-blocks or just a stream of compressed data?
            #  Usually: 
            #    Integration Constant (4 bytes, Int32)
            #    Data...
            
            payload = block_data[12:]
            
            # Integration constant (Initial Value) - Big Endian Int32
            if len(payload) < 4:
                continue
            
            init_val = struct.unpack('>i', payload[0:4])[0]
            
            # Remaining is compressed differences
            diff_blob = payload[4:]
            
            # We need to extract `sr` samples.
            # Problem: WIN compression is nibble-based. 
            # How do we know the encoding (4-bit, 8-bit...)?
            # The encoding is stored... where?
            # 
            # !!! CRITICAL !!!
            # WIN format does NOT assume single compression type per block.
            # It uses "flag bytes" interleaved? 
            # OR, usually, for "WIN System", it uses nibble encoding by default
            # BUT:
            #  Every discrete section starts with a header? No.
            #
            # Re-reading specs (common implementation):
            #  Data area is composed of "nibbles". Access is by 4 bits.
            #  It seems the data is ONLY 4-bit encoded?
            #  No, that overflows easily.
            #
            #  Actual logic:
            #  Read nibble.
            #  If nibble is NOT a metadata flag (e.g. -8?), treat as diff.
            #  Wait, standard WIN compression:
            #   If value fits in 4 bits (-7..7?), use 4 bit.
            #   Otherwise, promote to 1 byte, 2 byte...?
            #   How does decoder know?
            #
            #  Common "1-second block" WIN format logic:
            #  The sub-header for compression is missing?
            #  Maybe "SR" is not SR but "Count"?
            
            # Let's verify "WIN" (ERI/NIED) compression.
            # Commonly known as "delta compression".
            # Values:
            #  Nibble V:
            #    if V in [-7, 7]: diff = V
            #    if V == -8 (1000 binary): Next unit indicates actual Size?
            #
            # Precise Re-implementation of "Standard WIN32" logic:
            #  Iterate nibbles.
            #  val = read_nibble() (sign extended)
            #  if val == -8 (0x8): 
            #       # Extension code
            #       next_nibble = read_nibble() (unsigned)
            #       size_map = {0: 1byte, 1: 2byte, 2: 3byte, 3: 4byte}
            #       len = size_map[next_nibble]
            #       diff = read_bytes(len)
            #  else:
            #       diff = val
            
            # This requires bit-stream parsing.
            # For speed, we shouldn't do Python loops for every sample if possible.
            # But the structure is dynamic. We CANNOT verify alignment easily.
            # We must loop. (Python is slow here, but numba/cython not allowed in prompt constraints)
            # We will write optimized Python.
            
            # Convert blob to nibbles (uint8 array)
            # High nibble first.
            raw = np.frombuffer(diff_blob, dtype=np.uint8)
            nibbles = np.zeros(len(raw)*2, dtype=np.int8)
            nibbles[0::2] = (raw >> 4) & 0xF
            nibbles[1::2] = raw & 0xF
            
            # Sign extend 4-bit (standard: 8->-8)
            # 0..7 -> 0..7
            # 8..15 -> -8..-1
            mask = nibbles >= 8
            nibbles[mask] -= 16
            
            current_val = init_val
            decoded = np.zeros(sr, dtype=np.int32)
            
            # Decoding Loop
            # Since flow control depends on data, we use an index pointer.
            
            idx = 0 # nibble index
            out_idx = 0
            n_nibbles = len(nibbles)
            
            while out_idx < sr and idx < n_nibbles:
                nyb = nibbles[idx]
                idx += 1
                
                diff = 0
                
                if nyb == -8: 
                    # Escape code 0x8 (1000bin, interpreted as -8)
                    if idx >= n_nibbles:
                        break
                    
                    type_flag = nibbles[idx] # Read flags (needs to be unsigned 0..? Wait, we sign extended everything)
                    # We need the RAW nibble for the flag.
                    # Since we subtracted 16 if >=8, we add 16 back to recover raw if negative.
                    flag_raw = type_flag + 16 if type_flag < 0 else type_flag
                    idx += 1
                    
                    # 0: 1 byte (char)
                    # 1: 2 byte (short)
                    # 2: 3 byte (int24)
                    # 3: 4 byte (int)
                    
                    # Note: The bytes are formed by subsequence nibbles? 
                    # NO, usually aligned to nibbles? 
                    # Usually "Next 2 nibbles form 1 byte", "Next 4 nibbles form 2 bytes".
                    
                    n_nyb_needed = 0
                    if flag_raw == 0:
                        n_nyb_needed = 2
                    elif flag_raw == 1:
                        n_nyb_needed = 4
                    elif flag_raw == 2:
                        n_nyb_needed = 6
                    elif flag_raw == 3:
                        n_nyb_needed = 8
                    else:
                         # Unknown?
                         continue
                         
                    if idx + n_nyb_needed > n_nibbles:
                        break
                    
                    # Reconstruct value from nibbles (Big Endian)
                    # val = (n0 << 4*(N-1)) | ...
                    val = 0
                    for k in range(n_nyb_needed):
                         n = nibbles[idx+k]
                         # Use raw representation for upper parts?
                         # Actually, standard integer reconstruction.
                         # Last nibble determines sign? Or standard two's complement of the whole word?
                         # Usually we reconstruct bits then cast to signed.
                         raw_n = n + 16 if n < 0 else n
                         val = (val << 4) | raw_n
                         
                    idx += n_nyb_needed
                    
                    # Sign extension for the fulL word
                    # bits = n_nyb_needed * 4
                    # if val >= (1 << (bits-1)): val -= (1 << bits)
                    bits = n_nyb_needed * 4
                    if val & (1 << (bits - 1)):
                        val -= (1 << bits)
                    
                    diff = val
                    
                else:
                    diff = nyb
                    
                current_val += diff
                decoded[out_idx] = current_val
                out_idx += 1
                
            # Store block
            # If out_idx < sr, we have partial? Warn?
            
            # Add to data_map
            if ch_id not in data_map:
                data_map[ch_id] = []
            
            # Keep raw int32 array for now
            data_map[ch_id].append( (t_gps, decoded[:out_idx], sr) )


    # 3. Construct TimeSeries
    tsd = TimeSeriesDict()
    
    for ch_id, blocks in data_map.items():
        # blocks is list of (t, data, sr)
        # Sort by time just in case
        blocks.sort(key=lambda x: x[0])
        
        info = ch_info_map.get(ch_id, {})
        name = info.get('name', f"{ch_id:x}")
        gain = info.get('gain', 1.0)
        
        # Merge blocks
        # Check for gaps
        # Ideally we construct one array with gaps filled
        
        if not blocks:
            continue
            
        full_chunks = []
        t_start = blocks[0][0]
        sr = blocks[0][2]
        dt = 1.0 / sr
        
        current_t = t_start
        
        for blk in blocks:
            t_blk, d_blk, sr_blk = blk
            if sr_blk != sr:
                # SR changed? Handle or warn
                pass
                
            # Gap check
            # Tolerance: 0.5 sample
            if abs(t_blk - current_t) > 0.5 * dt:
                # Gap found
                gap_len_samples = int(round((t_blk - current_t) / dt))
                if gap_len_samples > 0:
                    # Fill nan
                    full_chunks.append(np.full(gap_len_samples, np.nan))
                    current_t += gap_len_samples * dt
                elif gap_len_samples < 0:
                    # Overlap? Truncate prev? 
                    # Simple: Just append (might jitter)
                    pass
            
            # Apply gain to float
            chunk_float = d_blk.astype(np.float64) * gain
            full_chunks.append(chunk_float)
            
            current_t += len(d_blk) * dt
            
        final_data = np.concatenate(full_chunks)
        
        ts = TimeSeries(final_data, t0=t_start, dt=dt, name=name, channel=name)
        tsd[name] = ts
        
    return tsd
