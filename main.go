package main

// #include <stdlib.h>
// #include <stdint.h>
/*
void start(const int device_id, const int threads, const int blocks, uint32_t *prefix, size_t difficulty, char *share_chunk, size_t share_difficulty, char *device_name, float *hashrate, unsigned char **out);
#cgo LDFLAGS: -L. -L./ -lkernel
*/
import "C"

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"github.com/btcsuite/btcutil/base58"
	"log"
	"math/rand"
	"strings"
	"sync"
	"time"
	"unsafe"
)

const devAddress = "DnAmhfPcckW4yDCVdaMQtPs6CkSfsQNyDrJ6kZzanJpty"

var (
	address, nodeUrl, poolUrl string
	deviceId, threads, blocks int
	silent, verbose           bool

	shareDifficulty int
	shares          = 0

	devFee            int // 1 every X shares are sent to the dev
	devFeeMustProcess = false

	res        MiningInfoResult
	deviceName = "Waiting..."
	hashrate   C.float

	addressesCache sync.Map
)

func main() {
	flag.StringVar(&address, "address", "", "denaro address (https://t.me/DenaroCoinBot)")
	flag.StringVar(&nodeUrl, "node", "https://denaro-node.gaetano.eu.org/", "denaro node url")
	flag.StringVar(&poolUrl, "pool", "https://denaro-pool.gaetano.eu.org/", "denaro pool url")

	flag.BoolVar(&silent, "silent", false, "silent mode (no output)")
	flag.BoolVar(&verbose, "verbose", false, "verbose mode (debug output)")

	flag.IntVar(&deviceId, "device", 0, "gpu device id")
	flag.IntVar(&threads, "threads", 512, "gpu threads")
	flag.IntVar(&blocks, "blocks", 50, "gpu blocks")

	flag.IntVar(&shareDifficulty, "share", 8, "share difficulty")
	flag.IntVar(&devFee, "fee", 5, "dev fee (1 every X shares are sent to the dev)")

	flag.Parse()

	// ask for address if not inserted as flag
	if len(address) == 0 {
		fmt.Print("Insert your address (available at https://t.me/DenaroCoinBot): ")
		if _, err := fmt.Scan(&address); err != nil {
			panic(err)
		}
	}

	getMiningInfo()

	// multi address
	addresses := strings.Split(address, ",")

	for {
		go printUI()

		address = addresses[rand.Intn(len(addresses))]
		miner(getMiningAddress(address))
	}
}

func printUI() {
	if !silent {
		if !verbose {
			// clear screen
			fmt.Print("\033[H\033[2J")
		}
		fmt.Printf(
			"Denaro GPU Miner\n\nDevice: %s\nAddress: %s\nHashrate: %.2f GH/s\n\nPool: %s\nNode: %s\n\nShares: %d\nDev fee: 1 share every %d shares\n\nLast update: %s\n",
			deviceName,
			address,
			float32(hashrate),
			poolUrl,
			nodeUrl,
			shares,
			devFee,
			time.Now().Format("15:04:05"),
		)
	}
}

func miner(miningAddress string) {
	var difficulty = res.Difficulty
	var idifficulty = int(difficulty)

	defer func() {
		if r := recover(); r != nil {
			log.Printf("Recovered from: %v\n", r)
		}
	}()

	lastBlock := res.LastBlock
	if lastBlock.Hash == "" {
		var num uint32 = 30_06_2005

		data := make([]byte, 32)
		binary.LittleEndian.PutUint32(data, num)

		lastBlock.Hash = hex.EncodeToString(data)
	}

	chunk := lastBlock.Hash[len(lastBlock.Hash)-idifficulty:]

	var shareChunk string

	if shareDifficulty > idifficulty {
		shareDifficulty = idifficulty
	}
	shareChunk = chunk[:shareDifficulty]

	var addressBytes []byte
	if devFeeMustProcess {
		addressBytes = stringToBytes(devAddress)
	} else {
		addressBytes = stringToBytes(miningAddress)
	}

	txs := res.PendingTransactionsHashes
	merkleTree := getTransactionsMerkleTree(txs)

	var prefix []byte
	// version, not supporting v1
	dataVersion := make([]byte, 2)
	binary.LittleEndian.PutUint16(dataVersion, uint16(2))
	prefix = append(prefix, dataVersion[0])

	dataHash, _ := hex.DecodeString(lastBlock.Hash)
	prefix = append(prefix, dataHash...)
	prefix = append(prefix, addressBytes...)
	dataMerkleTree, _ := hex.DecodeString(merkleTree)
	prefix = append(prefix, dataMerkleTree...)

	var shareChunkGpu = C.CString(shareChunk)

	sharesUChar := make([]*C.uchar, 32)
	for i := range sharesUChar {
		sharesUChar[i] = (*C.uchar)(C.malloc(108))
	}

	var deviceNameGpu = make([]byte, 256)

	C.start(
		C.int(deviceId),
		C.int(threads),
		C.int(blocks),
		(*C.uint)(unsafe.Pointer(&prefix[0])),
		C.ulong(difficulty*10),
		shareChunkGpu,
		C.ulong(shareDifficulty),
		(*C.char)(unsafe.Pointer(&deviceNameGpu[0])),
		(*C.float)(unsafe.Pointer(&hashrate)),
		&sharesUChar[0],
	)
	go postShares(sharesUChar)

	deviceName = C.GoString((*C.char)(unsafe.Pointer(&deviceNameGpu[0])))

	C.free(unsafe.Pointer(shareChunkGpu))
}

func getMiningAddress(address string) string {
	var reqP MiningAddress

	// use sync.Map as cache for addresses
	if cachedAddress, ok := addressesCache.Load(address); ok {
		return cachedAddress.(string)
	}

	for {
		req := GET(poolUrl+"get_mining_address", map[string]interface{}{"address": address})
		_ = json.Unmarshal(req.Body(), &reqP)

		if reqP.Ok {
			addressesCache.Store(address, reqP.Address)
			break
		} else {
			time.Sleep(1 * time.Second)
		}
	}
	return reqP.Address
}

func getMiningInfo() {
	var reqP MiningInfo

	for {
		req := GET(nodeUrl+"get_mining_info", map[string]interface{}{})
		_ = json.Unmarshal(req.Body(), &reqP)

		if reqP.Ok {
			res = reqP.Result
			break
		} else {
			time.Sleep(1 * time.Second)
		}
	}
}

func postShares(sharesUChar []*C.uchar) {
	var shareT Share

	for _, share := range sharesUChar {
		// check if first byte of result is 2, which currently is the version indicator
		if shareBytes := C.GoBytes(unsafe.Pointer(share), 108); shareBytes[0] == 2 {
			shareReq := POST(
				poolUrl+"share",
				map[string]interface{}{
					"block_content":    hex.EncodeToString(shareBytes),
					"txs":              res.PendingTransactionsHashes,
					"id":               res.LastBlock.Id + 1,
					"share_difficulty": res.Difficulty,
				},
			)
			_ = json.Unmarshal(shareReq.Body(), &shareT)

			// process dev fee
			devFeeText := ""
			if devFee > 0 && shares%devFee == 0 {
				devFeeMustProcess = true
			} else if devFeeMustProcess {
				devFeeMustProcess = false
				devFeeText = "(dev fee)"
			}

			if shareT.Ok {
				shares++

				if !silent {
					log.Printf("Share accepted (device: %d) %s\n", deviceId, devFeeText)
					log.Println(hex.EncodeToString(shareBytes))
				}
			} else {
				if !silent {
					log.Println(string(shareReq.Body()))
					log.Println(hex.EncodeToString(shareBytes))
				}
				getMiningInfo()
			}
		} else {
			break
		}
	}
}

func getTransactionsMerkleTree(transactions []string) string {

	var fullData []byte

	for _, transaction := range transactions {
		data, _ := hex.DecodeString(transaction)
		fullData = append(fullData, data...)
	}

	hash := sha256.New()
	hash.Write(fullData)

	return hex.EncodeToString(hash.Sum(nil))
}

func stringToBytes(text string) []byte {

	var data []byte

	data, err := hex.DecodeString(text)
	if err != nil {
		data = base58.Decode(text)
	}

	return data
}
