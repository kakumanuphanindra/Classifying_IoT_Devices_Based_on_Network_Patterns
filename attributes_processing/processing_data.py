import csv
import os
from scapy.all import *
from collections import defaultdict
from scapy.layers.tls.all import TLSClientHello, TLSServerHello

# Dictionary mapping MAC addresses to device names
mac_to_device = {
    "44:65:0d:56:cc:d3": "Amazon Echo",
    "e0:76:d0:3f:00:ae": "August Doorbell Cam",
    "70:88:6b:10:0f:c6": "Awair air quality monitor",
    "b4:75:0e:ec:e5:a9": "Belkin Camera",
    "ec:1a:59:83:28:11": "Belkin Motion Sensor",
    "ec:1a:59:79:f4:89": "Belkin Switch",
    "74:6a:89:00:2e:25": "Blipcare BP Meter",
    "7c:70:bc:5d:5e:dc": "Canary Camera",
    "30:8c:fb:2f:e4:b2": "Dropcam",
    "6c:ad:f8:5e:e4:61": "Google Chromecast",
    "28:c2:dd:ff:a5:2d": "Hello Barbie",
    "70:5a:0f:e4:9b:c0": "HP Printer",
    "74:c6:3b:29:d7:1d": "iHome PowerPlug",
    "d0:73:d5:01:83:08": "LiFX Bulb",
    "18:b4:30:25:be:e4": "NEST Smoke Sensor",
    "70:ee:50:18:34:43": "Netatmo Camera",
    "70:ee:50:03:b8:ac": "Netatmo Weather station",
    "00:17:88:2b:9a:25": "Phillip Hue Lightbulb",
    "e0:76:d0:33:bb:85": "Pixstart photo frame",
    "88:4a:ea:31:66:9d": "Ring Door Bell",
    "00:16:6c:ab:6b:88": "Samsung Smart Cam",
    "d0:52:a8:00:67:5e": "Smart Things",
    "f4:f2:6d:93:51:f1": "TP-Link Camera",
    "50:c7:bf:00:56:39": "TP-Link Plug",
    "18:b7:9e:02:20:44": "Triby Speaker",
    "00:24:e4:10:ee:4c": "Withings Baby Monitor",
    "00:24:e4:1b:6f:96": "Withings Scale",
    "00:24:e4:20:28:c6": "Withings sleep sensor",
    "00:24:e4:11:18:a8": "Withings"
}

# Function to check if an IP address is in the local network (192.168.1.*)
def is_local_ip(ip):
	return ip.startswith('192.168.1.')

# Updated function to extract cipher suite used for each flow
# def extract_cipher_suite(pkt,i):
	
	cipher_suite = "NONE"
	# Check for server chosen cipher in TLSServerHello
	if pkt.haslayer(TLSServerHello):
		# print("has server")
		# print(i)
		server_cipher = str(pkt[TLSServerHello].cipher)
		# print(server_cipher)
		if server_cipher == "None":
			cipher_suite = "NONE"
		else:
			server_cipher = server_cipher.strip('[]')
			# Split the string into individual numbers, then convert them to integers
			server_cipher_list = list(map(int, server_cipher.split()))
			# Output the resulting list
			# print(server_cipher_list)
			if server_cipher_list:
				# print(i)
				# print("in-server {pkt[TLSServerHello]}")
				cipher_suite = str(server_cipher_list[0])

	# Check for client preferred cipher in TLSClientHello
	elif pkt.haslayer(TLSClientHello):
		# print("has clinet")
		client_ciphers = str(pkt[TLSClientHello].ciphers)
		# print(client_ciphers)
		if client_ciphers == "None":
			cipher_suite = "NONE"
		else:
			client_ciphers = client_ciphers.strip('[]').replace(',', '')
			# Split the string into individual numbers, then convert them to integers
			client_cipher_list = list(map(int, client_ciphers.split()))
			# Output the resulting list
			# print(client_cipher_list)
			if client_cipher_list:
				# print("in-client {client_ciphers[0]}")
				# Access the first cipher in the list as a readable string
				cipher_suite = str(client_cipher_list[0])

	# print(cipher_suite)
	return cipher_suite

def extract_cipher_suite(pkt):
	cipher_suite = None
    # cipher_suite_list = ""
    # # Open the pcap file with a display filter for SSL/TLS handshake packets
    # # cap = pyshark.FileCapture(pcap_file, display_filter="ssl.handshake.type == 2")

    # packets = rdpcap(pcap_file)

    # Loop through each packet to find the Server Hello packet with cipher suite
    # print(len(cap))
	if pkt.haslayer(TLSServerHello):
		server_cipher = str(pkt[TLSServerHello].cipher)
		# print(server_cipher)
		if server_cipher == "None":
			cipher_suite = "NONE"
		else:
			server_cipher = server_cipher.strip('[]')
			# Split the string into individual numbers, then convert them to integers
			server_cipher_list = list(map(int, server_cipher.split()))
			# Output the resulting list
			if server_cipher_list:
				cipher_suite = str(server_cipher_list[0])

	elif pkt.haslayer(TLSClientHello):
		client_ciphers = str(pkt[TLSClientHello].ciphers)
		# print(client_ciphers)
		if client_ciphers == "None":
			cipher_suite = "NONE"
		else:
			client_ciphers = client_ciphers.strip('[]').replace(',', '')
			# Split the string into individual numbers, then convert them to integers
			client_cipher_list = list(map(int, client_ciphers.split()))
			if client_cipher_list:
				# Access the first cipher in the list as a readable string
				cipher_suite = str(client_cipher_list[0])

	return cipher_suite

# not using this medhod
def extract_dns_request(pkt):
	dns_request = None
	
	if pkt.haslayer(DNS) and pkt[DNS].qd is not None:
		# print(pkt.show)
		# Extract the query name (e.g., domain) from DNS query
		dns_request = str(pkt[DNS].qd.qname.decode()) # if pkt[DNS].qd.qname else "NONE"
		# print(str(pkt[DNS].qd.qname.decode()))
		# return dns_request
	return dns_request

# Function to extract flows with SYN and FIN packets from a PCAP file
def extract_flows_with_syn_fin(pcap_file):
	flows = defaultdict(list)  # To store flows
	flow_stats = []  # To store attributes of each flow

	packets = rdpcap(pcap_file)  # Load packets from PCAP file

	for pkt in packets:
		if IP in pkt:
			src_ip = pkt[IP].src
			dst_ip = pkt[IP].dst
			src_mac = pkt.src  # Extract source MAC address
			dst_mac = pkt.dst  # Extract destination MAC address

			# Skip flows that start and end with local devices
			if is_local_ip(src_ip) and is_local_ip(dst_ip):
				continue

			# dns_request = "NONE"
			# cipher_suite = "NONE"

			# Extract cipher suite and DNS request if available
			i=0
			i+=1
			cipher_suite = extract_cipher_suite(pkt)
			print(cipher_suite)
			dns_request = extract_dns_request(pkt)
			# print(dns_request)
			# if pkt.haslayer(DNS) and pkt[DNS].qd is not None:
			# 	# print(pkt.show)
			# 	# Extract the query name (e.g., domain) from DNS query
			# 	dns_request = str(pkt[DNS].qd.qname.decode())
			# 	print(dns_request)

			# # # Extract DNS request (if present)
			# # if pkt.haslayer(DNS) and pkt[DNS].qr == 0:  # DNS query
			# #     dns_request = pkt[DNS].qd.qname.decode() if pkt[DNS].qd else "NONE"

			# # # Extract cipher suite from TLS handshake (if present)
			# # if pkt.haslayer(TLSClientHello):
			# #     # Cipher suite from ClientHello
			# #     cipher_suite = pkt[TLSClientHello].ciphers[0].name if pkt[TLSClientHello].ciphers else "NONE"
			# # elif pkt.haslayer(TLSServerHello):
			# #     # Cipher suite from ServerHello
			# #     cipher_suite = pkt[TLSServerHello].cipher_suite.name if pkt[TLSServerHello].cipher_suite else "NONE"

			# Extract cipher suite if this is a TLS packet
			# cipher_suite = extract_cipher_suite(pkt)


			if TCP in pkt:
				src_port = pkt[TCP].sport
				dst_port = pkt[TCP].dport
				flow_key = (src_ip, src_port, dst_ip, dst_port)
				flow_key_reverse = (dst_ip, dst_port, src_ip, src_port)

				# Detect flow start using SYN
				if pkt[TCP].flags & 0x02:  # SYN flag
					flows[flow_key].append({'pkt': pkt, 'start_time': pkt.time, 'src_mac': src_mac, 'dst_mac': dst_mac})

				# Detect flow end using FIN or RST
				elif pkt[TCP].flags & 0x01 or pkt[TCP].flags & 0x04:  # FIN or RST flag
					if flow_key in flows:
						flows[flow_key].append({'pkt': pkt, 'end_time': pkt.time, 'src_mac': src_mac, 'dst_mac': dst_mac})
					elif flow_key_reverse in flows:
						flows[flow_key_reverse].append({'pkt': pkt, 'end_time': pkt.time, 'src_mac': src_mac, 'dst_mac': dst_mac})
			elif UDP in pkt:
				# Handle UDP packets
				src_port = pkt[UDP].sport
				dst_port = pkt[UDP].dport
				flow_key = (src_ip, src_port, dst_ip, dst_port)
				flows[flow_key].append({'pkt': pkt, 'start_time': pkt.time, 'src_mac': src_mac, 'dst_mac': dst_mac})
			else:
				# Handle non-TCP/UDP packets
				src_port = None
				dst_port = None
				protocol = pkt[IP].proto
				flow_key = (src_ip, src_port, dst_ip, dst_port, protocol)
				flows[flow_key].append({'pkt': pkt, 'start_time': pkt.time, 'src_mac': src_mac, 'dst_mac': dst_mac})

	# Process each flow to calculate attributes
	for flow_key, flow_packets in flows.items():
		start_time = min(pkt['start_time'] for pkt in flow_packets if 'start_time' in pkt)

		# If no end time is found, use the last packet time as end time
		end_time = max((pkt['end_time'] for pkt in flow_packets if 'end_time' in pkt), default=start_time)

		if start_time == end_time:
				end_time += 0.000001  # Small increment to avoid 0 duration

		# Calculate flow volume (sum of packet lengths)
		flow_volume = sum(len(pkt['pkt']) for pkt in flow_packets)

		# Filter out flows with volume less than 100 bytes
		# if flow_volume < 100:
		#     continue

		# Calculate flow duration
		flow_duration = end_time - start_time

		# Average flow rate = flow volume / duration
		average_flow_rate = flow_volume / flow_duration if flow_duration > 0 else 0

		# Extract MAC addresses
		src_mac = flow_packets[0]['src_mac']
		dst_mac = flow_packets[0]['dst_mac']

		# Find the device name based on source MAC
		device_name = mac_to_device.get(src_mac, "Unknown")

		# Append flow stats
		flow_stats.append({
			'src_ip': flow_key[0],
			'src_port': flow_key[1],
			'dst_ip': flow_key[2],
			'dst_port': flow_key[3],
			'dns_request': dns_request,
			'protocol': flow_key[4] if len(flow_key) > 4 else "TCP/UDP",
			'src_mac': src_mac,
			'dst_mac': dst_mac,
			'flow_volume': flow_volume,
			'flow_duration': flow_duration,
			'average_flow_rate': average_flow_rate,
			'start_time': start_time,
			'end_time': end_time,
			'cipher_suite': cipher_suite,
			'device_name': device_name
		})

	return flow_stats

# Function to calculate device sleep time
def calculate_device_sleep_time(flow_stats):
    device_flows = defaultdict(list)

    # Organize flows by source IP
    for flow in flow_stats:
        device_flows[flow['src_ip']].append(flow)

    # Calculate sleep time for each device
    for ip, flows in device_flows.items():
        # Sort flows by start time
        flows.sort(key=lambda x: x['start_time'])

        for i in range(len(flows)):
            current_flow = flows[i]
            if i < len(flows) - 1:
                next_flow = flows[i + 1]
                
                # Check if the flows overlap
                if next_flow['start_time'] <= current_flow['end_time']:
                    # Overlapping flows: set sleep time to 0
                    flows[i]['sleep_time'] = 0
                else:
                    # Calculate sleep time as time difference between flows
                    flows[i]['sleep_time'] = next_flow['start_time'] - current_flow['end_time']
            else:
                # No next flow, mark sleep time as -1
                flows[i]['sleep_time'] = -1

    updated_flows = []
    for ip_flows in device_flows.values():
        updated_flows.extend(ip_flows)

    return updated_flows

# Updated save_to_csv function with the additional filter
def save_to_csv(flow_stats, csv_file):
    fieldnames = [
        'src_ip', 'src_mac', 'src_port', 'dst_ip', 'dst_mac', 'dst_port', 'dns_request', 'protocol',
        'flow_volume', 'flow_duration', 'average_flow_rate', 'start_time', 'end_time', 'sleep_time', 'cipher_suite', 'device_name'
    ]
    
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for flow in flow_stats:
            # Check if source IP is local
            if is_local_ip(flow['src_ip']):
                writer.writerow(flow)

def analyze_pcap_folder(folder_path, csv_file):
    combined_flow_stats = []

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".pcap"):
            pcap_path = os.path.join(folder_path, filename)
            print(f"Processing {filename}...")
            flow_stats = extract_flows_with_syn_fin(pcap_path)
            combined_flow_stats.extend(flow_stats)

    # Calculate device sleep time for combined flows
    updated_flows = calculate_device_sleep_time(combined_flow_stats)
    save_to_csv(updated_flows, csv_file)

if __name__ == "__main__":
    pcap_folder = 'pcap_files'  # Folder containing the PCAP files
    csv_output = "all_attr_output.csv"  # CSV output file

    analyze_pcap_folder(pcap_folder, csv_output)
